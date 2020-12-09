use core::ops::AddAssign;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_stats::DeviationExt;
use num_traits::{cast::ToPrimitive, sign::Signed};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::time::{Duration, Instant};

use dtw::*;

// Working main function to try with naive implementation
/*fn main() {
    let (series_1, series_2) = utilities::make_test_series();
    let cost = naive::dtw(&series_1, &series_2, l2_dist, true);
    println!("cost: {}", cost);
}*/

/// Main Function
fn main() {
    // Input parameter

    let data_file_name = "Data.txt";
    let query_file_name = "Query.txt";
    let query_length_limit: Option<usize> = None; // Disregard everything of the query after this length
    let warping_window_size = 0.05;

    /*FILE *fp;            /// data file pointer
    FILE *qp;            /// query file pointer
    double bsf;          /// best-so-far
    double *u, *l, *qo, *uo, *lo,*tz,*u_d, *l_d;


    double d;
    long long i , j;
    double t1,t2;
    double dist=0, lb_kim=0, lb_k=0, lb_k2=0;
    */

    // Set limit to m if one was given
    let m = if let Some(limit) = query_length_limit {
        limit
    } else {
        usize::MAX
    };

    // data array and query array
    let mut t: Vec<f64> = Vec::new();
    let mut q: Vec<f64> = Vec::new();

    let best_so_far = f64::MAX;
    let (mut ex, mut ex2) = (0.0, 0.0);
    let (mut mean, mut std);
    let loc = 0;
    let kim = 0;
    let keogh = 0;
    let keogh2 = 0;

    // For every EPOCH points, all cummulative values, such as ex (sum), ex2 (sum square), will be restarted for reducing the floating point error.
    let EPOCH = 100000;

    // r  : size of Sakoe-Chiba warpping band
    let R: f64 = warping_window_size;
    let r: usize = if R <= 1.0 {
        (R * m as f64).floor() as usize
    } else {
        R.floor() as usize
    };

    let fp = File::open(data_file_name).unwrap(); // Open data file
    let qp = File::open(query_file_name).unwrap(); // Open query file

    // start the clock
    let t1 = Instant::now();

    // Read query file

    let mut buf = String::new(); // Single data point in sequence
    let mut reader = BufReader::new(qp);

    for _ in 0..m {
        if 0 == reader.read_line(&mut buf).unwrap() {
            break;
        }
        let d = buf.parse::<f64>().unwrap();
        ex += d;
        ex2 += d * d;
        q.push(d);
    }

    // Do z-normalize the query, keep in same array, q
    mean = ex / m as f64;
    std = ex2 / m as f64;
    std = f64::sqrt(std - mean * mean);
    for entry in q.iter_mut() {
        *entry = (*entry - mean) / std;
    }

    // Create envelop of the query: lower envelop, l, and upper envelop, u
    let (lower, upper) = ucr::lower_upper_lemire(&q, m, r);

    // Sort the query one time by abs(z-norm(q[i]))
    let mut q_tmp: Vec<(usize, f64)> = Vec::new();
    for (no, entry) in q.iter().enumerate() {
        q_tmp.push((no, *entry));
    }
    q_tmp.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // also create another arrays for keeping sorted envelop
    let mut order: Vec<usize> = Vec::new(); //new order of the query
    let mut qo: Vec<f64> = Vec::new();
    let mut uo: Vec<f64> = Vec::new();
    let mut lo: Vec<f64> = Vec::new();

    for i in 0..m {
        let (o, _) = q_tmp[i];
        order[i] = o;
        qo[i] = q[o];
        uo[i] = upper[o];
        lo[i] = lower[o];
    }

    // Initial the cummulative lower bound
    let cb = vec![0; m];
    let cb1 = cb.clone();
    let cb2 = cb.clone();

    let i = 0; // current index of the data in current chunk of size EPOCH
    let j = 0; // the starting index of the data in the circular array, t
    let (mut ex, mut ex2) = (0.0, 0.0);
    let done = false;
    let (it, ep, k) = (0, 0, 0);
    let index; // the starting index of the data in current chunk of size EPOCH
    let (buffer, u_buff, l_buff);

    while !done {
        // Read first m-1 points
        ep = 0;
        if it == 0 {
            for k in 0..(m - 1) {
                if fscanf(fp, "%lf", &d) != EOF {
                    buffer[k] = d;
                }
            }
        } else {
            for k in 0..m - 1 {
                buffer[k] = buffer[EPOCH - m + 1 + k];
            }
        }
    }
    /*



        // Read buffer of size EPOCH or when all data has been read.
        ep=m-1;
        while(ep<EPOCH)
        {   if (fscanf(fp,"%lf",&d) == EOF)
                break;
            buffer[ep] = d;
            ep++;
        }

        // Data are read in chunk of size EPOCH.
        // When there is nothing to read, the loop is end.
        if (ep<=m-1)
        {   done = true;
        } else
        {   lower_upper_lemire(buffer, ep, r, l_buff, u_buff);

            // Just for printing a dot for approximate a million point. Not much accurate.
            if (it%(1000000/(EPOCH-m+1))==0)
                fprintf(stderr,".");

            // Do main task here..
            ex=0; ex2=0;
            for(i=0; i<ep; i++)
            {
                // A bunch of data has been read and pick one of them at a time to use
                d = buffer[i];

                // Calcualte sum and sum square
                ex += d;
                ex2 += d*d;

                // t is a circular array for keeping current data
                t[i%m] = d;

                // Double the size for avoiding using modulo "%" operator
                t[(i%m)+m] = d;

                // Start the task when there are more than m-1 points in the current chunk
                if( i >= m-1 )
                {
                    mean = ex/m;
                    std = ex2/m;
                    std = sqrt(std-mean*mean);

                    // compute the start location of the data in the current circular array, t
                    j = (i+1)%m;
                    // the start location of the data in the current chunk
                    I = i-(m-1);

                    // Use a constant lower bound to prune the obvious subsequence
                    lb_kim = lb_kim_hierarchy(t, q, j, m, mean, std, bsf);

                    if (lb_kim < bsf)
                    {
                        // Use a linear time lower bound to prune; z_normalization of t will be computed on the fly.
                        // uo, lo are envelop of the query.
                        lb_k = lb_keogh_cumulative(order, t, uo, lo, cb1, j, m, mean, std, bsf);
                        if (lb_k < bsf)
                        {
                            // Take another linear time to compute z_normalization of t.
                            // Note that for better optimization, this can merge to the previous function.
                            for(k=0;k<m;k++)
                            {   tz[k] = (t[(k+j)] - mean)/std;
                            }

                            // Use another lb_keogh to prune
                            // qo is the sorted query. tz is unsorted z_normalized data.
                            // l_buff, u_buff are big envelop for all data in this chunk
                            lb_k2 = lb_keogh_data_cumulative(order, tz, qo, cb2, l_buff+I, u_buff+I, m, mean, std, bsf);
                            if (lb_k2 < bsf)
                            {
                                // Choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
                                // Note that cb and cb2 will be cumulative summed here.
                                if (lb_k > lb_k2)
                                {
                                    cb[m-1]=cb1[m-1];
                                    for(k=m-2; k>=0; k--)
                                        cb[k] = cb[k+1]+cb1[k];
                                }
                                else
                                {
                                    cb[m-1]=cb2[m-1];
                                    for(k=m-2; k>=0; k--)
                                        cb[k] = cb[k+1]+cb2[k];
                                }

                                // Compute DTW and early abandoning if possible
                                dist = dtw(tz, q, cb, m, r, bsf);

                                if( dist < bsf )
                                {   // Update bsf
                                    // loc is the real starting location of the nearest neighbor in the file
                                    bsf = dist;
                                    loc = (it)*(EPOCH-m+1) + i-m+1;
                                }
                            } else
                                keogh2++;
                        } else
                            keogh++;
                    } else
                        kim++;

                    // Reduce obsolute points from sum and sum square
                    ex -= t[j];
                    ex2 -= t[j]*t[j];
                }
            }

            // If the size of last chunk is less then EPOCH, then no more data and terminate.
            if (ep<EPOCH)
                done=true;
            else
                it++;
        }
    }

    i = (it)*(EPOCH-m+1) + ep;
    fclose(fp);

    free(q);
    free(u);
    free(l);
    free(uo);
    free(lo);
    free(qo);
    free(cb);
    free(cb1);
    free(cb2);
    free(tz);
    free(t);
    free(l_d);
    free(u_d);
    free(l_buff);
    free(u_buff);

    t2 = clock();
    printf("\n");

    // Note that loc and i are long long.
    cout << "Location : " << loc << endl;
    cout << "Distance : " << sqrt(bsf) << endl;
    cout << "Data Scanned : " << i << endl;
    cout << "Total Execution Time : " << (t2-t1)/CLOCKS_PER_SEC << " sec" << endl;

    // printf is just easier for formating ;)
    printf("\n");
    printf("Pruned by LB_Kim    : %6.2f%%\n", ((double) kim / i)*100);
    printf("Pruned by LB_Keogh  : %6.2f%%\n", ((double) keogh / i)*100);
    printf("Pruned by LB_Keogh2 : %6.2f%%\n", ((double) keogh2 / i)*100);
    printf("DTW Calculation     : %6.2f%%\n", 100-(((double)kim+keogh+keogh2)/i*100));
    return 0;*/
}

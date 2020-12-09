#[macro_use]
extern crate log;

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
    // TODO: There is an error with setting m, r and R correctly!!!

    // Initialize logger
    pretty_env_logger::init();

    // Input parameters
    let data_file_name = "Data.txt";
    let query_file_name = "Query2.txt";
    let query_length_limit: Option<usize> = Some(128); // Disregard everything of the query after this length
    let warping_window_size = 0.05;

    // Set limit to m if one was given
    let m = if let Some(limit) = query_length_limit {
        limit
    } else {
        usize::MAX
    };

    // Check boundaries of warping window
    let warping_window_size: f64 = if warping_window_size < 0.0 {
        0.0
    } else if warping_window_size > 1.0 {
        1.0
    } else {
        warping_window_size
    };

    // data array and query array
    let mut t: Vec<f64> = Vec::new();
    let mut tz: Vec<f64> = Vec::new();
    let mut q: Vec<f64> = Vec::new();

    let mut best_so_far = f64::MAX;
    let (mut ex, mut ex2) = (0.0, 0.0);
    let (mut mean, mut std);
    let mut loc = 0;
    let mut kim = 0;
    let mut keogh = 0;
    let mut keogh2 = 0;

    // For every EPOCH points, all cummulative values, such as ex (sum), ex2 (sum square), will be restarted for reducing the floating point error.
    let EPOCH = 100000;

    let fp = File::open(data_file_name).unwrap(); // Open data file
    let qp = File::open(query_file_name).unwrap(); // Open query file

    // start the clock
    let time_start = Instant::now();

    // Read query file
    let mut buf = String::new(); // Single data point in sequence
    let mut reader = BufReader::new(qp);

    for _ in 0..m {
        if 0 == reader.read_line(&mut buf).unwrap() {
            break;
        }
        //debug!("Try parsing {} from query to f64", buf.trim());
        let d = buf.trim().parse::<f64>().unwrap();
        buf.clear();
        ex += d;
        ex2 += d * d;
        q.push(d);
    }
    // Set length to query length
    let m = q.len();

    // r  : size of Sakoe-Chiba warpping band
    let r: usize = (warping_window_size * m as f64).floor() as usize;

    debug!("Done parsing query");
    //debug!("Query: {:?}", q);
    debug!("ex: {}", ex);
    debug!("ex2: {}", ex2);
    debug!("buf: {}", buf);

    // Do z-normalize the query, keep in same array, q
    mean = ex / m as f64;
    std = ex2 / m as f64;
    std = f64::sqrt(std - mean * mean);
    let q: Vec<f64> = q.iter_mut().map(|entry| (*entry - mean) / std).collect();
    debug!("Done z-normalizing the query");

    // Create envelope of the query: lower envelop, l, and upper envelop, u
    let (lower, upper) = ucr::lower_upper_lemire(&q, r);
    debug!("Created envelope of the query");

    debug!("Sort query");
    // Sort the query one time by abs(z-norm(q[i]))
    let mut q_tmp: Vec<(usize, f64)> = Vec::new();
    for (no, entry) in q.iter().enumerate() {
        q_tmp.push((no, *entry));
    }
    q_tmp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    debug!("Done sorting");

    // also create another arrays for keeping sorted envelop
    let mut order: Vec<usize> = vec![0; m]; //new order of the query
    let mut qo: Vec<f64> = vec![0.0; m];
    let mut uo: Vec<f64> = qo.clone();
    let mut lo: Vec<f64> = qo.clone();

    for i in 0..m {
        let (o, _) = q_tmp[i];
        order[i] = o;
        qo[i] = q[o];
        uo[i] = upper[o];
        lo[i] = lower[o];
    }

    // Initial the cummulative lower bound
    let mut cb = vec![0.0; m];
    let mut cb1 = cb.clone();
    let mut cb2 = cb.clone();

    let mut i = 0; // current index of the data in current chunk of size EPOCH
    let mut j = 0; // the starting index of the data in the circular array, t
    let (mut ex, mut ex2) = (0.0, 0.0);
    let mut done = false;
    let (mut it, mut ep, k) = (0, 0, 0);
    let mut index; // the starting index of the data in current chunk of size EPOCH
                   // In the C program it is called I
    let mut buffer: Vec<f64> = Vec::new();
    let mut u_buff: Vec<f64> = Vec::new();
    let mut l_buff: Vec<f64> = Vec::new();

    while !done {
        // Read first m-1 points
        if it == 0 {
            for _ in 0..(m - 1) {
                if 0 == reader.read_line(&mut buf).unwrap() {
                    break;
                }
                let d = buf.trim().parse::<f64>().unwrap();
                buf.clear();
                buffer.push(d);
            }
        } else {
            for k in 0..m - 1 {
                buffer[k] = buffer[EPOCH - m + 1 + k];
            }
        }

        // Read buffer of size EPOCH or when all data has been read.
        ep = m - 1;
        while ep < EPOCH {
            if 0 == reader.read_line(&mut buf).unwrap() {
                break;
            }
            let d = buf.trim().parse::<f64>().unwrap();
            buf.clear();
            buffer[ep] = d;
            ep += 1;
        }

        // Data are read in chunk of size EPOCH.
        // When there is nothing to read, the loop is end.
        if ep < m {
            done = true;
        } else {
            let (l_buff, u_buff) = ucr::lower_upper_lemire(&buffer, r);

            // Just for printing a dot for approximate a million point. Not much accurate.
            if it % (1000000 / (EPOCH - m + 1)) == 0 {
                print!(".");
            }

            // Do main task here..
            let (mut ex, mut ex2) = (0.0, 0.0);
            for i in 0..ep {
                // A bunch of data has been read and pick one of them at a time to use
                let d = buffer[i];

                // Calcualte sum and sum square
                ex += d;
                ex2 += d * d;

                // t is a circular array for keeping current data
                t[i % m] = d;

                // Double the size for avoiding using modulo "%" operator
                t[(i % m) + m] = d;

                // Start the task when there are more than m-1 points in the current chunk
                if i >= m - 1 {
                    mean = ex / m as f64;
                    std = ex2 / m as f64;
                    std = f64::sqrt(std - mean * mean);

                    // compute the start location of the data in the current circular array, t
                    j = (i + 1) % m;
                    // the start location of the data in the current chunk
                    index = i - (m - 1);

                    // Use a constant lower bound to prune the obvious subsequence
                    let lb_kim = ucr::lb_kim_hierarchy(&t, &q, j, m, mean, std);

                    if lb_kim < best_so_far {
                        // Use a linear time lower bound to prune; z_normalization of t will be computed on the fly.
                        // uo, lo are envelopes of the query.
                        let lb_k = ucr::lb_keogh_cumulative(
                            &order,
                            &t,
                            &uo,
                            &lo,
                            &mut cb1[..],
                            j,
                            m,
                            mean,
                            std,
                        );
                        if lb_k < best_so_far {
                            // Take another linear time to compute z_normalization of t.
                            // Note that for better optimization, this can merge to the previous function.
                            for k in 0..m {
                                tz[k] = (t[(k + j)] - mean) / std;
                            }

                            // Use another lb_keogh to prune
                            // qo is the sorted query. tz is unsorted z_normalized data.
                            // l_buff, u_buff are big envelopes for all data in this chunk
                            let lb_k2 = ucr::lb_keogh_data_cumulative(
                                &order,
                                &qo,
                                &mut cb2[..],
                                &l_buff[index..],
                                &u_buff[index..],
                                m,
                                mean,
                                std,
                            );
                            if lb_k2 < best_so_far {
                                // Choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
                                // Note that cb and cb2 will be cumulative summed here.
                                if lb_k > lb_k2 {
                                    cb[m - 1] = cb1[m - 1];
                                    for k in (0..m - 1).rev() {
                                        cb[k] = cb[k + 1] + cb1[k];
                                    }
                                } else {
                                    cb[m - 1] = cb2[m - 1];
                                    for k in (0..m - 1).rev() {
                                        cb[k] = cb[k + 1] + cb2[k];
                                    }
                                }

                                // Compute DTW and early abandoning if possible
                                let dist = ucr::dtw(&tz, &q, &cb, m, r);

                                if dist < best_so_far {
                                    // Update best_so_far
                                    // loc is the real starting location of the nearest neighbor in the file
                                    best_so_far = dist;
                                    loc = (it) * (EPOCH - m + 1) + i - m + 1;
                                }
                            } else {
                                keogh2 += 1;
                            }
                        } else {
                            keogh += 1;
                        }
                    } else {
                        kim += 1;
                    }

                    // Reduce obsolute points from sum and sum square
                    ex -= t[j];
                    ex2 -= t[j] * t[j];
                }
            }

            // If the size of last chunk is less then EPOCH, then no more data and terminate.
            if ep < EPOCH {
                done = true;
            } else {
                it += 1;
            }
        }
    }
    i = (it) * (EPOCH - m + 1) + ep;
    let time_end = Instant::now();
    println!();

    // Note that loc and i are long long.
    println!("Location : {}", loc);
    println!("Distance : {:.2}", f64::sqrt(best_so_far));
    println!("Data Scanned : {}", i);
    println!(
        "Total Execution Time : {:?}",
        time_end.duration_since(time_start)
    );

    // printf is just easier for formating ;)
    println!();
    println!("Pruned by LB_Kim    : {:.2}", (kim / i) * 100);
    println!("Pruned by LB_Keogh  : {:.2}", (keogh / i) * 100);
    println!("Pruned by LB_Keogh2 : {:.2}", (keogh2 / i) * 100);
    println!(
        "DTW Calculation     : {:.2}",
        100 - ((kim + keogh + keogh2) / i * 100)
    );
}

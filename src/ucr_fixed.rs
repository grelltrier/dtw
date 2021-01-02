use ndarray::prelude::*;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::time::Instant;

use super::*;

fn dist(x: f64, y: f64) -> f64 {
    (x - y).powi(2)
}

pub struct Trillion;

impl Trillion {
    fn upper_lower_lemire(query: &[f64], len: usize, r: usize) -> (Vec<f64>, Vec<f64>) {
        let mut upper = vec![0.0; len];
        let mut lower = upper.clone();
        let mut du: VecDeque<usize> = VecDeque::with_capacity(2 * r + 2);
        let mut dl: VecDeque<usize> = VecDeque::with_capacity(2 * r + 2);

        du.push_back(0);
        dl.push_back(0);
        for i in 1..len {
            if i > r {
                upper[i - r - 1] = query[*du.front().unwrap()];
                lower[i - r - 1] = query[*dl.front().unwrap()];
            }

            // Pop out the bound that is not maximium or minimum
            // Store the max upper bound and min lower bound within window r
            if query[i] > query[i - 1] {
                du.pop_back();
                while !du.is_empty() && query[i] > query[*du.back().unwrap()] {
                    du.pop_back();
                }
            } else {
                dl.pop_back();
                while !dl.is_empty() && query[i] < query[*dl.back().unwrap()] {
                    dl.pop_back();
                }
            }
            du.push_back(i);
            dl.push_back(i);

            // i - r - 1 == r + du.first
            // Pop out the bound that os out of window r.
            if i == 2 * r + 1 + du.front().unwrap() {
                du.pop_front();
            } else if i == 2 * r + 1 + dl.front().unwrap() {
                dl.pop_front();
            }
        }

        // The envelop of first r points are from r+1 .. r+r, so the last r points' envelop haven't settle down yet.
        for i in len..(len + r + 1) {
            upper[i - r - 1] = query[*du.front().unwrap()];
            lower[i - r - 1] = query[*dl.front().unwrap()];
            if i - du.front().unwrap() >= 2 * r + 1 {
                du.pop_front();
            }
            if i - dl.front().unwrap() >= 2 * r + 1 {
                dl.pop_front();
            }
        }
        (lower, upper)
    }

    pub fn lb_kim_hierarchy(
        t: &[f64],
        q: &[f64],
        j: usize,
        mean: f64,
        std: f64,
        bsf: Option<f64>,
    ) -> (f64, usize) {
        let len = q.len();
        let bsf = bsf.unwrap_or(f64::INFINITY);

        let mut d;
        let mut lb;

        // 1 point at front and back
        let x0 = (t[j] - mean) / std;
        let y0 = (t[(j + len - 1)] - mean) / std;
        lb = dist(x0, q[0]) + dist(y0, q[len - 1]);
        if lb >= bsf {
            return (lb, 1);
        }

        // 2 points at front
        let x1 = (t[(j + 1)] - mean) / std;
        d = f64::min(dist(x1, q[0]), dist(x0, q[1]));
        d = f64::min(d, dist(x1, q[1]));
        lb += d;
        if d >= bsf {
            return (lb, 2);
        }
        if lb >= bsf {
            return (lb, 1);
        }

        // 2 points at back
        let y1 = (t[(j + len - 2)] - mean) / std;
        d = f64::min(dist(y1, q[len - 1]), dist(y0, q[len - 2]));
        d = f64::min(d, dist(y1, q[len - 2]));
        lb += d;
        if d >= bsf {
            return (lb, len - 2 + 1);
        }
        if lb >= bsf {
            return (lb, 1);
        }

        // 3 points at front
        let x2 = (t[(j + 2)] - mean) / std;
        d = f64::min(dist(x0, q[2]), dist(x1, q[2]));
        d = f64::min(d, dist(x2, q[2]));
        d = f64::min(d, dist(x2, q[1]));
        d = f64::min(d, dist(x2, q[0]));
        lb += d;
        if d >= bsf {
            return (lb, 3);
        }
        if lb >= bsf {
            return (lb, 1);
        }
        // 3 points at back
        let y2 = (t[(j + len - 3)] - mean) / std;
        d = f64::min(dist(y0, q[len - 3]), dist(y1, q[len - 3]));
        d = f64::min(d, dist(y2, q[len - 3]));
        d = f64::min(d, dist(y2, q[len - 2]));
        d = f64::min(d, dist(y2, q[len - 1]));
        lb += d;
        if d >= bsf {
            return (lb, len - 3 + 1);
        }

        (lb, 1)
    }

    /// LB_Keogh 1: Create Envelop for the query
    /// Note that because the query is known, envelop can be created once at the begining.
    ///
    /// Variable Explanation,
    /// order : sorted indices for the query.
    /// t     : a circular array keeping the current data.
    /// uo, lo: upper and lower envelops for the query, which already sorted.
    /// j     : index of the starting location in t
    /// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
    pub fn lb_keogh_cumulative(
        order: &[usize],
        t: &[f64],
        uo: &[f64],
        lo: &[f64],
        cb: &mut [f64],
        j: usize,
        len: usize,
        mean: f64,
        std: f64,
        bsf: Option<f64>,
    ) -> (f64, usize) {
        let bsf = bsf.unwrap_or(f64::INFINITY);

        let mut lb: f64 = 0.0;
        let mut jump = order[0];
        let mut x;
        let mut d;

        for i in 0..len {
            x = (t[(order[i] + j)] - mean) / std;
            d = 0.0;

            if order[i] < jump {
                jump = order[i]
            }

            if x > uo[i] {
                d = dist(x, uo[i]);
            } else if x < lo[i] {
                d = dist(x, lo[i]);
            }
            lb += d;
            cb[order[i]] = d;

            if lb >= bsf {
                break;
            }
        }
        (lb, jump + 1)
    }

    /// LB_Keogh 2: Create Envelop for the data
    /// Note that the envelops have been created (in main function) when each data point has been read.
    ///
    /// Variable Explanation,
    /// tz: Z-normalized data
    /// qo: sorted query
    /// cb: (output) current bound at each position. Used later for early abandoning in DTW.
    /// l,u: lower and upper envelop of the current data
    pub fn lb_keogh_data_cumulative(
        order: &[usize],
        qo: &[f64],
        cb: &mut [f64],
        l: &[f64],
        u: &[f64],
        j: usize, // The C implementation calls this 'len'
        mean: f64,
        std: f64,
        bsf: Option<f64>,
    ) -> (f64, usize) {
        let bsf = bsf.unwrap_or(f64::INFINITY);

        let mut uu;
        let mut ll;
        let mut d;

        let mut lb = 0.0;
        let mut jump = order[0];

        // TODO: The C implementation repeats this loop 'len' times
        for i in 0..order.len() {
            uu = (u[j + order[i]] - mean) / std;
            ll = (l[j + order[i]] - mean) / std;
            d = 0.0;

            if order[i] < jump {
                jump = order[i]
            }

            if qo[i] > uu {
                d = dist(qo[i], uu);
            } else if qo[i] < ll {
                d = dist(qo[i], ll);
            }
            lb += d;
            cb[order[i]] = d;
            if lb >= bsf {
                break;
            }
        }
        (lb, jump + 1)
    }

    /// Calculate Dynamic Time Wrapping distance
    /// A,B: data and query, respectively
    /// cb : cummulative bound used for early abandoning
    /// r  : size of Sakoe-Chiba warpping band
    pub fn dtw(
        a_seq: &[f64],
        b_seq: &[f64],
        cb: &[f64],
        //m: usize, // This argument is added in the C implementation
        r: usize,
        bsf: Option<f64>,
    ) -> f64 {
        let bsf = bsf.unwrap_or(f64::INFINITY);
        let mut cost_tmp;
        let (mut x, mut y, mut z, mut min_cost);
        let a_seq_len = a_seq.len();

        // Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
        let mut cost = Array::<f64, Ix1>::from_elem(2 * r + 1, f64::INFINITY);
        let mut cost_prev = cost.clone();
        let mut k = 0;

        // TODO: The C implementation calls this m times
        for i in 0..a_seq_len {
            k = r.saturating_sub(i);
            min_cost = f64::INFINITY;

            for j in i.saturating_sub(r)..(usize::min(a_seq_len - 1, i + r) + 1) {
                // Initialize all row and column
                if (i == 0) && (j == 0) {
                    cost[k] = dist(a_seq[0], b_seq[0]); // TODO: Calculation of dist should be possible to move out of the loop since it is independent of i and j
                    min_cost = cost[k];
                    k += 1; // TODO: This should be necessary, double check with C implementation
                    continue;
                }

                if (j < 1) || (k < 1) {
                    y = f64::INFINITY;
                } else {
                    y = cost[k - 1];
                }
                if (i < 1) || (k + 1 > 2 * r) {
                    x = f64::INFINITY;
                } else {
                    x = cost_prev[k + 1];
                }
                if (i < 1) || (j < 1) {
                    z = f64::INFINITY;
                } else {
                    z = cost_prev[k];
                }

                // Classic DTW calculation
                cost[k] = f64::min(f64::min(x, y), z) + dist(a_seq[i], b_seq[j]);

                // Find minimum cost in row for early abandoning (possibly to use column instead of row).
                if cost[k] < min_cost {
                    min_cost = cost[k];
                }
                k += 1;
            }

            // We can abandon early if the current cummulative distace with lower bound together are larger than bsf
            if i + r < a_seq_len - 1 && min_cost + cb[i + r + 1] >= bsf {
                return min_cost + cb[i + r + 1]; // TODO: C implementation returns the bsf as well
            }

            // Move current array to previous array.
            cost_tmp = cost;
            cost = cost_prev;
            cost_prev = cost_tmp;
        }
        k -= 1;

        // the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
        cost_prev[k] // TODO: C implementation returns the bsf as well
    }

    /// Print function for debugging
    pub fn printArray(array: &[f64]) {
        for cell in array.iter() {
            print!("{:>6.2}", cell);
        }
        println!();
    }

    pub fn calculate(
        data_name: &str,
        query_name: &str,
        window_rate: f64,
        sort: bool,
        dont_jump: bool,
    ) {
        let EPOCH = 100000;
        let mut loc = 0;
        let (mut jump_times, mut kim, mut keogh, mut keogh2) = (0, 0, 0, 0);
        let (mut ex, mut ex2) = (0.0, 0.0);
        let mut bsf = f64::INFINITY;

        // Variables not initialized in Ruby implementation
        let mut q: Vec<f64> = Vec::new();

        // Initialized by Ruby implementation
        //dist = lb_kim = lb_k = lb_k2 = 0.0
        //i = j = 0

        // start the clock
        let time_start = Instant::now();

        // Open file containing query and creating things to read from it
        let qp = File::open(query_name).unwrap(); // Open query file
        let mut queryreader = BufReader::new(qp);
        let mut buf = String::new(); // Buffer to read to

        // Read all lines of query
        while queryreader.read_line(&mut buf).unwrap() != 0 {
            let d = buf.trim().parse::<f64>().unwrap();
            buf.clear();
            ex += d;
            ex2 += d.powi(2);
            q.push(d);
        }

        // Do z-normalize the query, keep in same array, q
        let mut mean = ex / q.len() as f64;
        let mut std = f64::sqrt((ex2 / q.len() as f64) - mean.powi(2));
        let q: Vec<f64> = q.iter_mut().map(|entry| (*entry - mean) / std).collect();

        // TODO: This is done differently in C implementation, double check it
        // r  : size of Sakoe-Chiba warpping band
        let r = if window_rate <= 1.0 {
            (window_rate * q.len() as f64).floor()
        } else {
            window_rate.floor()
        };
        let r = r as usize;

        // Create envelope of the query: lower envelop, l, and upper envelop, u
        let (l, u) = Self::upper_lower_lemire(&q, q.len(), r);

        // Add the index to each query point
        let mut q_tmp: Vec<(usize, f64)> = Vec::new();
        for (i, q_point) in q.iter().enumerate() {
            q_tmp.push((i, *q_point));
        }

        // Create more arrays for keeping the (sorted) envelop
        let mut order: Vec<usize> = Vec::new();
        let mut qo: Vec<f64> = Vec::new();
        let mut uo: Vec<f64> = Vec::new();
        let mut lo: Vec<f64> = Vec::new();

        if sort {
            q_tmp.sort_by(|a, b| {
                (b.1.abs())
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(Ordering::Equal)
            });

            q_tmp.iter().for_each(|x| {
                order.push(x.0);
                qo.push(q[x.0]);
                uo.push(u[x.0]);
                lo.push(l[x.0]);
            })
        } else {
            for i in 0..q.len() {
                order.push(i);
            }
            qo = q.clone();
            uo = u.clone();
            lo = l.clone();
        }

        // Initial the cummulative lower bound
        let mut cb = vec![0.0; q.len()];
        let mut cb1 = cb.clone();
        let mut cb2 = cb.clone();

        let (mut i, mut j) = (0, 0); // i: the current index of the data in current chunk of size EPOCH
                                     // j: the starting index of the data in the circular array t
        let (mut ex, mut ex2) = (0.0, 0.0);
        let mut done = false;
        let mut it = 0;
        let mut ep = 0;
        // let k = 0; Never used

        let mut buffer: Vec<f64> = vec![0.0; EPOCH];
        let mut t: Vec<f64> = vec![0.0; q.len() * 2];
        let mut tz: Vec<f64> = vec![0.0; q.len()]; // TODO: The Ruby implementation requests an array of size 'q.len()*2'
                                                   // Looks like this can be omitted completely

        // Create a reader/iterator to get the data from
        let mut data_container = utilities::DataContainer::new(data_name);

        // ################# UP TO HERE EVERY VARIABLE IS THE SAME BETWEEN RUBY AND RUST ########################

        while !done {
            ep = 0;
            // Read first m-1 points
            if it == 0 {
                for k in 0..(q.len() - 1) {
                    if let Some(data) = data_container.next() {
                        buffer[k] = data;
                    } /*else {
                          break;
                      }*/
                }
            } else {
                for k in 0..q.len() - 1 {
                    buffer[k] = buffer[EPOCH - q.len() + 1 + k];
                }
            }

            // Read buffer of size EPOCH or when all data has been read.
            ep = q.len() - 1;
            while ep < EPOCH {
                if let Some(data) = data_container.next() {
                    buffer[ep] = data;
                    ep += 1;
                } else {
                    break;
                }
            }

            if ep < q.len() {
                done = true;
            } else {
                let (l_buff, u_buff) = Self::upper_lower_lemire(&buffer, ep, r);

                // Just for printing a dot for approximate a million point. Not much accurate.
                if it % (1000000 / (EPOCH - q.len() + 1)) == 0 {
                    print!(".");
                }

                let (mut ex, mut ex2) = (0.0, 0.0);
                let mut jump = 0;

                // Do main task here..
                for i in 0..ep {
                    // A bunch of data has been read and pick one of them at a time to use
                    let d = buffer[i];

                    // Calcualte sum and sum square
                    ex += d;
                    ex2 += d.powi(2);

                    // t is a circular array for keeping current data
                    t[i % q.len()] = d;

                    // Double the size for avoiding using modulo "%" operator
                    t[(i % q.len()) + q.len()] = d;

                    if jump > 0 {
                        jump -= 1
                    }

                    // Start the task when there are more than m-1 points in the current chunk
                    if i >= q.len() - 1 {
                        // compute the start location of the data in the current circular array, t
                        j = (i + 1) % q.len();

                        if dont_jump || jump <= 0 {
                            mean = ex / q.len() as f64;
                            std = f64::sqrt((ex2 / q.len() as f64) - mean.powi(2));

                            // the start location of the data in the current chunk
                            let i_cap = i - (q.len() - 1);

                            // Use a constant lower bound to prune the obvious subsequence
                            let (lb_kim, jump) =
                                Self::lb_kim_hierarchy(&t, &q, j, mean, std, Some(bsf));

                            //////
                            if lb_kim < bsf {
                                let (lb_k, jump) = Self::lb_keogh_cumulative(
                                    &order,
                                    &t,
                                    &uo,
                                    &lo,
                                    &mut cb1[..],
                                    j,
                                    q.len(),
                                    mean,
                                    std,
                                    Some(bsf),
                                );

                                if lb_k < bsf {
                                    let (lb_k2, jump) = Trillion::lb_keogh_data_cumulative(
                                        &order,
                                        &qo,
                                        &mut cb2[..],
                                        &l_buff, // Maybe pass '&l_buff[index..]' instead
                                        &u_buff, // Maybe pass '&u_buff[index..]' instead
                                        i_cap,
                                        mean,
                                        std,
                                        Some(bsf),
                                    );

                                    if lb_k2 < bsf {
                                        {
                                            // Choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
                                            // Note that cb and cb2 will be cumulative summed here.
                                            if lb_k > lb_k2 {
                                                cb[q.len() - 1] = cb1[q.len() - 1];
                                                for k in (0..q.len() - 1).rev() {
                                                    cb[k] = cb[k + 1] + cb1[k];
                                                }
                                            } else {
                                                cb[q.len() - 1] = cb2[q.len() - 1];
                                                for k in (0..q.len() - 1).rev() {
                                                    cb[k] = cb[k + 1] + cb2[k];
                                                }
                                            }

                                            // Take another linear time to compute z_normalization of t.
                                            // Note that for better optimization, this can merge to the previous function.

                                            let tz: Vec<f64> = t
                                                .iter_mut()
                                                .skip(j)
                                                .take(q.len())
                                                .map(|entry| (*entry - mean) / std)
                                                .collect();

                                            let dist = Self::dtw(&tz, &q, &cb, r, Some(bsf));
                                            //////
                                            ///
                                            if dist < bsf {
                                                // Update bsf
                                                // loc is the real starting location of the nearest neighbor in the file
                                                bsf = dist;
                                                loc = it * (EPOCH - q.len() + 1) + i + 1 - q.len();
                                            }
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
                        } else {
                            jump_times += 1
                        }
                        // Reduce obsolute points from sum and sum square
                        ex -= t[j];
                        ex2 -= t[j].powi(2);
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

        let time_end = Instant::now();
        println!();

        // Note that loc and i are long long.
        println!("Location : {}", loc);
        println!("Distance : {}", f64::sqrt(bsf));
        i = it * (EPOCH - q.len() + 1) + ep;
        println!("Data Scanned : {}", i);
        println!(
            "Total Execution Time : {:?}",
            time_end.saturating_duration_since(time_start)
        );

        // Convert to f64 so the following calculations are not rounded to 0
        let i = i as f64;
        let jump_times = jump_times as f64;
        let kim = kim as f64;
        let keogh = keogh as f64;
        let keogh2 = keogh2 as f64;

        println!();
        println!("Pruned by Jump      : {:.4}%", (jump_times / i) * 100.0);
        println!("Pruned by LB_Kim    : {:.4}%", (kim / i) * 100.0);
        println!("Pruned by LB_Keogh  : {:.4}%", (keogh / i) * 100.0);
        println!("Pruned by LB_Keogh2 : {:.4}%", (keogh2 / i) * 100.0);
        println!(
            "DTW Calculation     : {:.4}%",
            100.0 - ((jump_times + kim + keogh + keogh2) / i * 100.0)
        );

        if loc != 430264 {
            error!("Location should be 430264 but it is {}", loc);
        }
        if (3.790699169990 - f64::sqrt(bsf)).abs() > 0.000000000001 {
            error!("Distance should be 3.79069 but it is {:.4}", f64::sqrt(bsf));
        }
    }
}

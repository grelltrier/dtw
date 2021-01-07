// use ndarray::prelude::*;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::time::{Duration, Instant};

use super::*;
mod structs;

#[derive(Copy, Clone, Debug)]
pub struct Settings {
    jump: bool,
    sort: bool,
    normalize: bool,
    window_rate: f64,
    epoch: usize,
}
impl Settings {
    pub fn new(
        jump: bool,
        sort: bool,
        normalize: bool,
        window_rate: f64,
        epoch: usize,
    ) -> Settings {
        Settings {
            jump,
            sort,
            normalize,
            window_rate,
            epoch,
        }
    }
}
impl Default for Settings {
    fn default() -> Self {
        Settings {
            jump: true,
            sort: true,
            normalize: true,
            window_rate: 0.1,
            epoch: 100000,
        }
    }
}

fn dist(x: f64, y: f64) -> f64 {
    (x - y).powi(2)
}

// Calculates the absolute delta between the two values
fn delta(x: f64, y: f64) -> f64 {
    (x - y).abs()
}

// Calculates the minimum of the absolute delta between the values 'end' of each sequence and all previous values from the other sequence. If reverse is true, the sequences are reversed and the index is counted from the back to the front
fn min_delta(seq_a: (&f64, &[f64]), seq_b: (&f64, &[f64])) -> f64 {
    let mut result = delta(*seq_a.0, *seq_b.0);
    // If there are more then one element in the slice, compare their deltas with the values of the end of the slice
    if seq_a.1.len() > 0 {
        let sequences = [seq_a.1, seq_b.1];
        for value in seq_a.1 {
            result = f64::min(result, delta(*value, *seq_b.0));
        }
        for value in seq_b.1 {
            result = f64::min(result, delta(*value, *seq_a.0));
        }
    }
    result
}

#[derive(Copy, Clone, Debug)]
struct CalcResult {
    loc: usize,
    bsf: f64,
    duration: Duration,
    i: usize,
    supplemental_stats: Option<UCRStats>,
}

#[derive(Copy, Clone, Debug)]
struct UCRStats {
    jump_times: usize,
    kim: f64,
    keogh: f64,
    keogh2: f64,
}
impl UCRStats {
    fn print(&self, i: f64) {
        println!(
            "Pruned by Jump      : {:.4}%",
            (self.jump_times as f64 / i) * 100.0
        );
        println!("Pruned by LB_Kim    : {:.4}%", (self.kim / i) * 100.0);
        println!("Pruned by LB_Keogh  : {:.4}%", (self.keogh / i) * 100.0);
        println!("Pruned by LB_Keogh2 : {:.4}%", (self.keogh2 / i) * 100.0);
        println!(
            "DTW Calculation     : {:.4}%",
            100.0 - ((self.jump_times as f64 + self.kim + self.keogh + self.keogh2) / i * 100.0)
        );
    }
}

#[derive(Clone, Debug)]
pub struct Trillion {
    target: (String, String),
    result: Option<CalcResult>,
    settings: Settings,
}

impl Trillion {
    pub fn new(data_name: &str, query_name: &str, settings: Settings) -> Self {
        Self {
            target: (data_name.to_owned(), query_name.to_owned()),
            result: None,
            settings,
        }
    }
    pub fn print(&self) {
        if let Some(result) = self.result {
            println!();
            println!("Location : {}", result.loc);
            println!("Distance : {}", f64::sqrt(result.bsf));
            println!("Data Scanned : {}", result.i);
            println!("Total Execution Time : {:?}", result.duration);
            println!();

            // Print additional stats about pruning
            if let Some(stats) = result.supplemental_stats {
                stats.print(result.i as f64);
            }

            // This is just for testing
            // Throws an error to easily see when the calculation is wrong
            if result.loc != 430264 {
                error!("Location should be 430264 but it is {}", result.loc);
            }
            if (3.790699169990 - f64::sqrt(result.bsf)).abs() > 0.000000000001 {
                error!(
                    "Distance should be 3.79069 but it is {:.4}",
                    f64::sqrt(result.bsf)
                );
            }
        }
    }
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

    fn lb_kim_hierarchy(
        t: &[f64],
        q: &[f64],
        j: usize,
        mean: f64,
        std: f64,
        bsf: f64,
    ) -> (f64, usize) {
        let no_pruning_points = 3;

        let mut lb = 0.0;

        let mut end_idx;
        let mut end_value; // Stores the end of the sequence
                           // The end changes, depending on if the sequence is looked at front to back or back to front
        let mut range;

        let begin_idx = [0, q.len() - 1]; // The index from which the points are counted from. The first index is for the back and the second for the front
                                          // It is important to check the end first because it enables us to jump further

        let mut candidate_z = [Vec::new(), Vec::new()]; // First one is for the back, the second for the front

        for i in 0..no_pruning_points {
            for idx in 0..2 {
                // If the values are iterated over from the front to back, the index needs to be added
                // If the values are iterated over from the back to front, the index needs to be subtracted
                if idx == 0 {
                    end_idx = begin_idx[idx] + i;
                    range = 0..end_idx;
                } else {
                    end_idx = begin_idx[idx] - i;
                    range = end_idx + 1..end_idx + 1 + i;
                };
                end_value = (&t[j + end_idx] - mean) / std;
                let d =
                    min_delta((&end_value, &candidate_z[idx]), (&q[end_idx], &q[range])).powi(2);
                lb += d;
                if d >= bsf {
                    let jump_size = if idx == 0 { q.len() - i } else { 1 + i };
                    return (lb, jump_size);
                }
                if lb >= bsf {
                    return (lb, 1);
                }
                candidate_z[idx].push(end_value);
            }
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
    fn lb_keogh_cumulative(
        order: &[usize],
        data: &[f64],
        cum_bound: &mut [f64],
        upper_envelope: &[f64],
        lower_envelope: &[f64],
        j: usize,
        mean: f64,
        std: f64,
        bsf: f64,
        data_bound: bool,
    ) -> (f64, usize) {
        let mut q_z;
        let mut u_z;
        let mut l_z;
        let mut diff;

        let mut lb: f64 = 0.0;
        let mut jump = order[0];

        for i in 0..order.len() {
            if data_bound {
                q_z = data[i];
                u_z = (upper_envelope[j + order[i]] - mean) / std;
                l_z = (lower_envelope[j + order[i]] - mean) / std;
            } else {
                q_z = (data[(j + order[i])] - mean) / std;
                u_z = upper_envelope[i];
                l_z = lower_envelope[i];
            }
            diff = 0.0;

            if order[i] < jump {
                jump = order[i]
            }

            if q_z > u_z {
                diff = dist(u_z, q_z);
            } else if q_z < l_z {
                diff = dist(l_z, q_z);
            }
            lb += diff;
            cum_bound[order[i]] = diff;

            if lb >= bsf {
                break;
            }
        }
        (lb, jump + 1)
    }

    /// Calculate Dynamic Time Wrapping distance
    /// a_seq, b_seq: data and query, respectively
    /// cb : cummulative bound used for early abandoning
    /// r  : size of Sakoe-Chiba warpping band
    fn dtw(a_seq: &[f64], b_seq: &[f64], cb: &[f64], r: usize, bsf: f64) -> f64 {
        //let bsf = bsf.unwrap_or(f64::INFINITY);
        let mut cost_tmp;
        let (mut x, mut y, mut z, mut min_cost);
        let a_seq_len = a_seq.len();

        // Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(a_seq_len).
        let mut cost = Array::<f64, Ix1>::from_elem(a_seq_len, f64::INFINITY);
        let mut cost_prev = cost.clone();

        // Variables to implement the pruning - PrunedDTW
        let mut sc = 0;
        let mut ec = 0;
        let mut next_ec;
        let mut lp = 0; // lp stands for last pruning
                        // TODO: Should this be initialized to 0? UCR_USP_suite does not intialize it at all it seems?
        let mut ub = bsf - cb[r + 1];
        let mut found_sc: bool;
        let mut pruned_ec = false;
        let mut ini_j;

        for i in 0..a_seq_len {
            // k = r.saturating_sub(i);
            min_cost = f64::INFINITY;

            found_sc = false;
            pruned_ec = false;
            next_ec = i + r + 1;

            ini_j = usize::max(i.saturating_sub(r), sc);

            for j in ini_j..(usize::min(a_seq_len - 1, i + r) + 1) {
                // Initialize all row and column
                if (i == 0) && (j == 0) {
                    cost[j] = dist(a_seq[0], b_seq[0]);
                    min_cost = cost[j];
                    found_sc = true;
                    continue;
                }

                if j == ini_j {
                    y = f64::INFINITY;
                } else {
                    y = cost[j - 1];
                }
                if (i == 0) || (j == i + r) || (j >= lp) {
                    x = f64::INFINITY;
                } else {
                    x = cost_prev[j];
                }
                if (i == 0) || (j == 0) || (j > lp) {
                    z = f64::INFINITY;
                } else {
                    z = cost_prev[j - 1];
                }

                // Classic DTW calculation
                cost[j] = f64::min(f64::min(x, y), z) + dist(a_seq[i], b_seq[j]);

                // Find minimum cost in row for early abandoning (possibly to use column instead of row).
                if cost[j] < min_cost {
                    min_cost = cost[j];
                }

                // Pruning criteria
                if !found_sc && cost[j] <= ub {
                    sc = j;
                    found_sc = true;
                }

                if cost[j] > ub {
                    if j > ec {
                        lp = j;
                        pruned_ec = true;
                        break;
                    }
                } else {
                    next_ec = j + 1;
                }
            }

            if i + r < a_seq_len - 1 {
                ub = bsf - cb[i + r + 1];
                // We can abandon early if the current cummulative distace with lower bound together are larger than bsf
                if min_cost + cb[i + r + 1] >= bsf {
                    return f64::INFINITY;
                }
            }

            // Move current array to previous array.
            cost_tmp = cost;
            cost = cost_prev;
            cost_prev = cost_tmp;

            if sc > 0 {
                cost_prev[sc - 1] = f64::INFINITY;
            }

            if !pruned_ec {
                lp = i + r + 1;
            }

            ec = next_ec;
        }

        // If pruned in the last row
        if pruned_ec {
            cost_prev[a_seq_len - 1] = f64::INFINITY;
        }

        // the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
        cost_prev[a_seq_len - 1]
    }

    pub fn calculate(&mut self) {
        let (data_name, query_name) = &self.target;
        let Settings {
            window_rate,
            sort,
            normalize,
            jump,
            epoch,
        } = self.settings;

        let mut query: Vec<f64> = Vec::new();
        let mut loc = 0;
        let (mut jump_times, mut kim, mut keogh, mut keogh2) = (0, 0, 0, 0);
        let (mut ex, mut ex2) = (0.0, 0.0);
        let mut bsf = f64::INFINITY;

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
            query.push(d);
        }

        // Do z-normalize the query, keep in same array, q
        let mut mean = ex / query.len() as f64;
        let mut std = f64::sqrt((ex2 / query.len() as f64) - mean.powi(2));

        if normalize {
            query = query
                .iter_mut()
                .map(|entry| (*entry - mean) / std)
                .collect();
        }

        // TODO: This is done differently in C implementation, double check it
        // sakoe_chiba_band  : size of Sakoe-Chiba warpping band
        let sakoe_chiba_band = if window_rate <= 1.0 {
            (window_rate * query.len() as f64).floor() as usize
        } else {
            window_rate.floor() as usize
        };

        // Create envelope of the query
        let (lower_envelop, upper_envelop) =
            Self::upper_lower_lemire(&query, query.len(), sakoe_chiba_band);

        // Add the index to each query entry
        let mut indexed_query: Vec<(usize, f64)> = Vec::new();
        for (idx, query_entry) in query.iter().enumerate() {
            indexed_query.push((idx, *query_entry));
        }

        // Create more arrays for keeping the (sorted) envelop
        let mut order: Vec<usize> = Vec::new();
        let mut qo: Vec<f64> = Vec::new();
        let mut uo: Vec<f64> = Vec::new();
        let mut lo: Vec<f64> = Vec::new();

        if sort {
            indexed_query.sort_by(|a, b| {
                (b.1.abs())
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(Ordering::Equal)
            });

            indexed_query.iter().for_each(|x| {
                order.push(x.0);
                qo.push(query[x.0]);
                uo.push(upper_envelop[x.0]);
                lo.push(lower_envelop[x.0]);
            })
        } else {
            for i in 0..query.len() {
                order.push(i);
            }
            qo = query.clone();
            uo = upper_envelop;
            lo = lower_envelop;
        }

        // Initialize the cummulative lower bound
        let mut cb = vec![0.0; query.len()];
        let mut cb1 = cb.clone();
        let mut cb2 = cb.clone();

        let mut j; // j: the starting index of the data in the circular array t
        let mut done = false;
        let mut it = 0;
        let mut ep = 0;

        let mut buffer: Vec<f64> = vec![0.0; epoch];
        let mut t: Vec<f64> = vec![0.0; query.len() * 2];
        let mut tz: Vec<f64> = Vec::new(); // z-normalized candidate sequence
        tz.reserve(query.len());

        // Create a reader/iterator to get the data from
        let mut data_container = utilities::DataContainer::new(&data_name);

        while !done {
            // Read first m-1 points
            if it == 0 {
                for k in 0..(query.len() - 1) {
                    if let Some(data) = data_container.next() {
                        buffer[k] = data;
                    }
                }
            } else {
                for k in 0..(query.len() - 1) {
                    buffer[k] = buffer[epoch - query.len() + 1 + k];
                }
            }

            // Read buffer of size EPOCH or when all data has been read.
            ep = query.len() - 1;
            while ep < epoch {
                if let Some(data) = data_container.next() {
                    buffer[ep] = data;
                    ep += 1;
                } else {
                    break;
                }
            }

            if ep < query.len() {
                done = true;
            } else {
                let (l_buff, u_buff) = Self::upper_lower_lemire(&buffer, ep, sakoe_chiba_band);

                // Just for printing a dot for approximate a million point. Not much accurate.
                if it % (1000000 / (epoch - query.len() + 1)) == 0 {
                    print!(".");
                }

                ex = 0.0;
                ex2 = 0.0;
                let mut jump_size: usize = 0;

                // Do main task here..
                for i in 0..ep {
                    // A bunch of data has been read and pick one of them at a time to use
                    let data = buffer[i];

                    // Calcualte sum and sum square
                    ex += data;
                    ex2 += data.powi(2);

                    // t is a circular array for keeping current data
                    t[i % query.len()] = data;

                    // Double the size for avoiding using modulo "%" operator
                    t[(i % query.len()) + query.len()] = data;

                    jump_size = jump_size.saturating_sub(1);

                    // Start the task when there are more than m-1 points in the current chunk
                    if i >= query.len() - 1 {
                        // compute the start location of the data in the current circular array, t
                        j = (i + 1) % query.len();

                        if !jump || jump_size == 0 {
                            mean = ex / query.len() as f64;
                            std = f64::sqrt((ex2 / query.len() as f64) - mean.powi(2));

                            // the start location of the data in the current chunk
                            let i_cap = i - (query.len() - 1);

                            // Use a constant lower bound to prune the obvious subsequence
                            let (lb_kim, jump_size_tmp) =
                                Self::lb_kim_hierarchy(&t, &query, j, mean, std, bsf);
                            jump_size = jump_size_tmp;

                            //////
                            if lb_kim < bsf {
                                let (lb_keogh, jump_tmp) = Self::lb_keogh_cumulative(
                                    &order,
                                    &t,
                                    &mut cb1[..],
                                    &uo,
                                    &lo,
                                    j,
                                    mean,
                                    std,
                                    bsf,
                                    false,
                                );
                                jump_size = jump_tmp;

                                if lb_keogh < bsf {
                                    let (lb_keogh_data, jump_tmp) = Self::lb_keogh_cumulative(
                                        &order,
                                        &qo,
                                        &mut cb2[..],
                                        &u_buff, // Maybe pass '&u_buff[index..]' instead
                                        &l_buff, // Maybe pass '&l_buff[index..]' instead
                                        i_cap,
                                        mean,
                                        std,
                                        bsf,
                                        true,
                                    );
                                    jump_size = jump_tmp;

                                    if lb_keogh_data < bsf {
                                        {
                                            // Choose better lower bound between lb_keogh and lb_keogh2 to be used in early abandoning DTW
                                            // Note that cb and cb2 will be cumulative summed here.
                                            if lb_keogh > lb_keogh_data {
                                                cb[query.len() - 1] = cb1[query.len() - 1];
                                                for k in (0..query.len() - 1).rev() {
                                                    cb[k] = cb[k + 1] + cb1[k];
                                                }
                                            } else {
                                                cb[query.len() - 1] = cb2[query.len() - 1];
                                                for k in (0..query.len() - 1).rev() {
                                                    cb[k] = cb[k + 1] + cb2[k];
                                                }
                                            }

                                            // Take another linear time to compute z_normalization of t.
                                            // Note that for better optimization, this can merge to the previous function.
                                            if normalize {
                                                tz = t
                                                    .iter_mut()
                                                    .skip(j)
                                                    .take(query.len())
                                                    .map(|entry| (*entry - mean) / std)
                                                    .collect();
                                            }

                                            let dist =
                                                Self::dtw(&tz, &query, &cb, sakoe_chiba_band, bsf);

                                            if dist < bsf {
                                                // Update bsf
                                                // loc is the real starting location of the nearest neighbor in the file
                                                bsf = dist;
                                                loc = it * (epoch - query.len() + 1) + i + 1
                                                    - query.len();
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
                if ep < epoch {
                    done = true;
                } else {
                    it += 1;
                }
            }
        }

        let time_end = Instant::now();
        let duration = time_end.saturating_duration_since(time_start);
        let i = it * (epoch - query.len() + 1) + ep;
        let kim = kim as f64;
        let keogh = keogh as f64;
        let keogh2 = keogh2 as f64;

        let supplemental_stats = Some(UCRStats {
            jump_times,
            kim,
            keogh,
            keogh2,
        });
        self.result = Some(CalcResult {
            loc,
            bsf,
            duration,
            i,
            supplemental_stats,
        });
    }
}

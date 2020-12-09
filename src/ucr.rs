use std::collections::VecDeque;

use super::*;

fn dist(x: f64, y: f64) -> f64 {
    (x - y) * (x - y)
}

/// Finding the envelop of min and max value for LB_Keogh
/// Implementation idea is intoruduced by Danial Lemire in his paper
/// "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
pub fn lower_upper_lemire(query: &[f64], r: usize) -> (Vec<f64>, Vec<f64>) {
    let len = query.len();
    let mut upper = vec![0.0; len];
    let mut lower = upper.clone();

    debug!("r: {}", r);
    debug!("Initialize deque");
    let mut du: VecDeque<usize> = VecDeque::with_capacity(2 * r + 2);
    let mut dl: VecDeque<usize> = VecDeque::with_capacity(2 * r + 2);
    debug!("Initialized deque");

    du.push_back(0);
    dl.push_back(0);

    debug!("Start first loop");
    for i in 1..len {
        if i > r {
            debug!("Branch started: if i > r");
            upper[i - r - 1] = query[*du.front().unwrap()];
            lower[i - r - 1] = query[*dl.front().unwrap()];
            debug!("Branch ended: if i > r");
        }
        if query[i] > query[i - 1] {
            debug!("Branch started: query[i] > query[i - 1]");
            du.pop_back();
            while !du.is_empty() && query[i] > query[*du.back().unwrap()] {
                du.pop_back();
            }
            debug!("Branch ended: query[i] > query[i - 1]");
        } else {
            debug!("Branch started: else");
            dl.pop_back();
            while !dl.is_empty() && query[i] < query[*dl.back().unwrap()] {
                dl.pop_back();
            }
            debug!("Branch ended: else");
        }
        du.push_back(i);
        dl.push_back(i);

        if i == 2 * r + 1 + du.front().unwrap() {
            du.pop_front();
        } else if i == 2 * r + 1 + dl.front().unwrap() {
            dl.pop_front();
        }
    }
    debug!("First loop completed");
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

/// ########################################################################
/// Variable explanation from other methods:
/// order : sorted indices for the query.
/// uo, lo: upper and lower envelops for the query, which already sorted.
/// t     : a circular array keeping the current data.
/// j     : index of the starting location in t
/// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
/// ########################################################################

/// Calculate quick lower bound
/// Usually, LB_Kim take time O(m) for finding top,bottom,fist and last.
/// However, because of z-normalization the top and bottom cannot give siginifant benefits.
/// And using the first and last points can be computed in constant time.
/// The prunning power of LB_Kim is non-trivial, especially when the query is not long, say in length 128.
pub fn lb_kim_hierarchy(t: &[f64], q: &[f64], j: usize, len: usize, mean: f64, std: f64) -> f64 {
    let mut d;
    let mut lb;
    let best_so_far = f64::MAX;

    // 1 point at front and back
    let x0 = (t[j] - mean) / std;
    let y0 = (t[(len - 1 + j)] - mean) / std;
    lb = dist(x0, q[0]) + dist(y0, q[len - 1]);
    if lb >= best_so_far {
        return lb;
    }

    // 2 points at front
    let x1 = (t[(j + 1)] - mean) / std;
    d = f64::min(dist(x1, q[0]), dist(x0, q[1]));
    d = f64::min(d, dist(x1, q[1]));
    lb += d;
    if lb >= best_so_far {
        return lb;
    }

    // 2 points at back
    let y1 = (t[(len - 2 + j)] - mean) / std;
    d = f64::min(dist(y1, q[len - 1]), dist(y0, q[len - 2]));
    d = f64::min(d, dist(y1, q[len - 2]));
    lb += d;
    if lb >= best_so_far {
        return lb;
    }

    // 3 points at front
    let x2 = (t[(j + 2)] - mean) / std;
    d = f64::min(dist(x0, q[2]), dist(x1, q[2]));
    d = f64::min(d, dist(x2, q[2]));
    d = f64::min(d, dist(x2, q[1]));
    d = f64::min(d, dist(x2, q[0]));
    lb += d;
    if lb >= best_so_far {
        return lb;
    }

    // 3 points at back
    let y2 = (t[(len - 3 + j)] - mean) / std;
    d = f64::min(dist(y0, q[len - 3]), dist(y1, q[len - 3]));
    d = f64::min(d, dist(y2, q[len - 3]));
    d = f64::min(d, dist(y2, q[len - 2]));
    d = f64::min(d, dist(y2, q[len - 1]));
    lb += d;
    lb
}

/// LB_Keogh 1: Create Envelop for the query
/// Note that because the query is known, envelop can be created once at the begenining.
///
/// Variable Explanation,
/// order : sorted indices for the query.
/// uo, lo: upper and lower envelops for the query, which already sorted.
/// t     : a circular array keeping the current data.
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
) -> f64 {
    let mut lb: f64 = 0.0;
    let mut x;
    let mut d;
    let best_so_far = f64::MAX;

    for i in 0..len {
        if lb < best_so_far {
            x = (t[(order[i] + j)] - mean) / std;
            d = 0.0;
            if x > uo[i] {
                d = dist(x, uo[i]);
            } else if x < lo[i] {
                d = dist(x, lo[i]);
            }
            lb += d;
            cb[order[i]] = d;
        }
    }
    lb
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
    len: usize,
    mean: f64,
    std: f64,
) -> f64 {
    let mut lb = 0.0;
    let mut uu;
    let mut ll;
    let mut d;
    let best_so_far = f64::MAX;

    for i in 0..len {
        if lb < best_so_far {
            uu = (u[order[i]] - mean) / std;
            ll = (l[order[i]] - mean) / std;
            d = 0.0;
            if qo[i] > uu {
                d = dist(qo[i], uu);
            } else if qo[i] < ll {
                d = dist(qo[i], ll);
            }
            lb += d;
            cb[order[i]] = d;
        }
    }
    lb
}

/// Calculate Dynamic Time Wrapping distance
/// A,B: data and query, respectively
/// cb : cummulative bound used for early abandoning
/// r  : size of Sakoe-Chiba warpping band
pub fn dtw(A: &[f64], B: &[f64], cb: &[f64], m: usize, r: usize) -> f64 {
    let best_so_far = f64::MAX;
    let mut cost_tmp;
    let mut k = 0;
    let (mut x, mut y, mut z, mut min_cost);

    // Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).

    let mut cost = Array::<f64, Ix1>::from_elem(2 * r + 1, f64::MAX);
    let mut cost_prev = cost.clone();

    for i in 0..m {
        min_cost = f64::MAX;
        k = r.saturating_sub(i);

        let mut j = i.saturating_sub(r);
        while j <= usize::min(m - 1, i + r) {
            // Initialize all row and column
            if (i == 0) && (j == 0) {
                cost[k] = dist(A[0], B[0]);
                min_cost = cost[k];
                continue;
            }

            if (j < 1) || (k < 1) {
                y = f64::MAX;
            } else {
                y = cost[k - 1];
            }
            if (i < 1) || (k + 1 > 2 * r) {
                x = f64::MAX;
            } else {
                x = cost_prev[k + 1];
            }
            if (i < 1) || (j < 1) {
                z = f64::MAX;
            } else {
                z = cost_prev[k];
            }

            // Classic DTW calculation
            cost[k] = f64::min(f64::min(x, y), z) + dist(A[i], B[j]);

            // Find minimum cost in row for early abandoning (possibly to use column instead of row).
            if cost[k] < min_cost {
                min_cost = cost[k];
            }
            j += 1;
            k += 1;
        }

        // We can abandon early if the current cummulative distace with lower bound together are larger than best_so_far
        if i + r < m - 1 && min_cost + cb[i + r + 1] >= best_so_far {
            //free(cost);
            //free(cost_prev);
            return min_cost + cb[i + r + 1];
        }

        // Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k -= 1;

    // the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    //free(cost);
    //free(cost_prev);
    cost_prev[k]
}

/// Print function for debugging
pub fn printArray(array: &[f64]) {
    for cell in array.iter() {
        print!("{:>6.2}", cell);
    }
    println!();
}

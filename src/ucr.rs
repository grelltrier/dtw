use super::*;
use std::ops::{Div, Sub};

/// Calculate the Dynamic Time Wrapping distance
/// data, query: data and query time series, respectively
/// cb : cummulative bound used for early abandoning
/// r  : size of Sakoe-Chiba warpping band
/// bsf: The DTW of the current best match (used for abandoning)
/// cost_fn: Function to calculate the cost between observations
pub fn dtw<T, F>(data: &[T], query: &[T], cb: &[f64], r: usize, bsf: f64, cost_fn: &F) -> f64
where
    T: Div<Output = T> + Sub<Output = T>,
    F: Fn(&T, &T) -> f64,
{
    let mut cost_tmp;
    let (mut x, mut y, mut z, mut min_cost);
    let data_len = data.len();

    // Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(data_len).
    let mut cost = Array::<f64, Ix1>::from_elem(data_len, f64::INFINITY);
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

    for i in 0..data_len {
        min_cost = f64::INFINITY;

        found_sc = false;
        pruned_ec = false;
        next_ec = i + r + 1;

        ini_j = usize::max(i.saturating_sub(r), sc);

        for j in ini_j..(usize::min(data_len - 1, i + r) + 1) {
            // Initialize all row and column
            if (i == 0) && (j == 0) {
                cost[j] = cost_fn(&data[0], &query[0]);
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
            cost[j] = f64::min(f64::min(x, y), z) + cost_fn(&data[i], &query[j]);

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

        if i + r < data_len - 1 {
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
        cost_prev[data_len - 1] = f64::INFINITY;
    }

    // the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
    cost_prev[data_len - 1]
}

use super::*;
use std::ops::{Div, Sub};

/// Calculate the Dynamic Time Wrapping distance
/// data, query: data and query time series, respectively
/// cb : cummulative bound used for early abandoning
/// w  : size of Sakoe-Chiba warpping band
/// bsf: The DTW of the current best match (used for abandoning)
/// cost_fn: Function to calculate the cost between observations
pub fn dtw<T, F>(data: &[T], query: &[T], cb: &[f64], w: usize, bsf: f64, cost_fn: &F) -> f64
where
    T: Div<Output = T> + Sub<Output = T>,
    F: Fn(&T, &T) -> f64,
{
    let (mut x, mut y, mut z); // Top left, top, left cells next to currently calculated cell
    let data_len = data.len(); // Also called n

    // Instead of using matrix of size O(n^2) or O(n*w), we will reuse two array of size O(n).
    let mut cost = Array::<f64, Ix1>::from_elem(2 * w, f64::INFINITY);
    let mut cost_prev = cost.clone();
    let mut cost_tmp; // Used to switch cost to cost_prev

    let mut sc = 0;
    let mut ec = 0;
    let mut step_start = Some(0);
    let mut step_end = 2 * w;
    let mut end;

    let mut ub = bsf - cb[w];

    //let mut found_start = true;

    let mut warp_start;
    let mut warp_end;

    for i in 0..data_len {
        // If no start was found, abandon calculation
        if let Some(start) = sc_next {
            sc = start;
        } else {
            break;
        }

        // Reset the start for the next row
        sc_next = None;

        // Update the UB
        if i + w < data_len - 1 {
            ub = bsf - cb[i + w];
        }

        warp_start = i.saturating_sub(w);
        warp_end = i + w;

        sc = usize::max(sc, warp_start);
        end = usize::min(warp_end, data_len - 1);

        for j in sc..=end {
            //   |    x    |    y    |
            //   |    z    | current |

            // If current cell is at the leftmost column
            if j == sc {
                z = f64::INFINITY;
            } else {
                z = cost[j - 1];
            }

            if (i == 0) || (j == i + w) || (j >= ec_prev) {
                y = f64::INFINITY;
            } else {
                y = cost_prev[j];
            }
            if (i == 0) || (j == 0) || (j > ec_prev) {
                x = f64::INFINITY;
            } else {
                x = cost_prev[j - 1];
            }

            // Classic DTW calculation
            cost[j] = f64::min(f64::min(x, y), z) + cost_fn(&data[i], &query[j]);

            // If the cost is lower than UB ...
            if cost[j] < ub {
                // The row does not end here so we push the end to the next cell
                ec = j + 1;
                // If no start for the next column was previously found,
                // now we found one so save its index
                if sc_next.is_none() {
                    sc_next = Some(j);
                }
            // If the cost is greater or equal to the UB ...
            } else {
                // and if the column is greater than the end column of the previous row, the row must end
                if j > ec_prev {
                    // .. so we remember at which index we pruned and that we pruned the end
                    break;
                }
            }
        }

        // Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;

        ec_prev = ec;
    }
    if sc_next.is_none() {
        return f64::INFINITY;
    }

    cost_prev[sc - 1]
}

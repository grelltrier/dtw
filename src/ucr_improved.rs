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
    let debug = false; // TODO: REMOVE!!
    let mut cell_value; // TODO: REMOVE!!!
    let mut ub;

    let co;
    let li;
    if query.len() <= data.len() {
        co = query;
        li = data;
    } else {
        co = data;
        li = query;
    }

    let mut j; // Column index/index of the shorter sequence
    let mut c; // Cost to match observation i and j with each other

    // Instead of using matrix of size O(n^2) or O(n*w), we will reuse two array of size O(n).
    let mut prev = Array::<f64, Ix1>::from_elem(co.len() + 1, f64::INFINITY);
    let mut curr = prev.clone();
    let mut cost_tmp;

    curr[0] = 0.0;
    let mut next_start = 0;
    let mut prev_pruning_point = 0;
    let mut pruning_point = 0;

    let mut warp_band_begin;
    let mut warping_end;

    // For each row of the cost matrix
    'row_loop: for i in 0..li.len() {
        // Begin at the start column
        warp_band_begin = i.saturating_sub(w);
        if next_start < warp_band_begin {
            if next_start <= prev_pruning_point {
                j = warp_band_begin;
                next_start = warp_band_begin;
            } else {
                return f64::INFINITY;
            }
        } else {
            j = next_start;
        }
        warping_end = usize::min(i + w, co.len() - 1);

        if debug {
            println!();
            for _ in 0..j {
                print!("|  ");
            }
        }

        ub = bsf - cb[i];

        // Swap the current array with the previous array
        cost_tmp = curr;
        curr = prev;
        prev = cost_tmp;

        prev_pruning_point = pruning_point;

        // Set the value left of the start to infinity so we can always use that value
        curr[j] = f64::INFINITY; // This should be uneccessary

        // While we have not found a start (a cell with a lower cost than the UB)
        while j == next_start {
            // If the column is smaller than the previous_pruning_point,
            // we can have valid warping paths from the top and top left of the cell.
            // Since we have not found a start yet, all values left of the current cell
            // must exceed the UB so we don't bother looking at them
            if j < prev_pruning_point {
                c = cost_fn(&li[i], &co[j]);
                cell_value = c + f64::min(prev[j + 1], prev[j]);
            // If the column equals the previous_pruning_point,
            // we still have a chance to find a valid match but
            // the top left subsequence is the only possibility for a valid path
            } else if j == prev_pruning_point {
                c = cost_fn(&li[i], &co[j]);
                cell_value = c + prev[j];
            // If j > prev_pruning_point and we haven't found a start yet,
            // we can abandon the calculation
            } else {
                return f64::INFINITY;
            }
            if debug {
                print!("|{:>2}", cell_value);
            }
            curr[j + 1] = cell_value;
            // If the calculated sum of costs is lower than the UB,
            // then we found a valid match, so the pruning point of
            // this row must be further to the right. We also DON'T push the next_start to the right
            if curr[j + 1] <= ub {
                pruning_point = j + 1;
            // If the cost is higher, its not a valid match so we need to continue looking for a start
            // We push the next_start to the right
            } else {
                next_start += 1;
            }
            // Increase the column index
            j += 1;
            if j > warping_end {
                continue 'row_loop;
            }
        }
        // Once we found a start, we can now also have warping paths coming from the left
        // of the current cell, so we must consider that cell now too
        // While the column is lower than the previous_pruning_point, we can have warping
        // paths comming from the left, top and top right cells.
        // The previous_pruning_point can never be higher than co.len()+1 so we can not
        // exceed the cost matrix in this loop
        while j < prev_pruning_point {
            c = cost_fn(&li[i], &co[j]);
            cell_value = c + f64::min(curr[j], f64::min(prev[j + 1], prev[j]));
            if debug {
                print!("|{:>2}", cell_value);
            }
            curr[j + 1] = cell_value;
            if curr[j + 1] <= ub {
                pruning_point = j + 1;
            }
            j += 1;
            if j > warping_end {
                continue 'row_loop;
            }
        }
        // When reaching this point, we found a start and reached the prev_pruning_point,
        // but have not yet reached the end of the row.
        // The only possible warping paths are the left and top-left cells
        // At this point j = prev_pruning_point
        if j == prev_pruning_point && j < co.len() {
            c = cost_fn(&li[i], &co[j]);
            cell_value = c + f64::min(curr[j], prev[j]);
            if debug {
                print!("|{:>2}", cell_value);
            }
            curr[j + 1] = cell_value;
            if curr[j + 1] <= ub {
                pruning_point = j + 1;
            }
            j += 1;
            if j > warping_end {
                continue 'row_loop;
            }
        }
        // Once we passed the prev_pruning_point, the only possible warping paths are from the left
        // and we can start with the next row once we found a value that is greater than the UB
        while j < co.len() {
            c = cost_fn(&li[i], &co[j]);
            cell_value = c + curr[j];
            if debug {
                print!("|{:>2}", cell_value);
            }
            curr[j + 1] = cell_value;
            if curr[j + 1] <= ub {
                pruning_point = j + 1;
            } else {
                break; // Breaks the while loop
            }
            j += 1;
            if j > warping_end {
                continue 'row_loop;
            }
        }
    }
    // The boundary constraint dictates, that the last points must match.
    // This only is the case, if the previous_pruning_point is pushed all the way to the right
    if prev_pruning_point == co.len() {
        curr[co.len()]
    } else {
        return f64::INFINITY;
    }
}

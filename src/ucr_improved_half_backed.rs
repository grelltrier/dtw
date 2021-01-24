use super::*;
use std::ops::{Div, Sub};

/// Calculate the Dynamic Time Wrapping distance
/// data, query: data and query time series, respectively
/// cb : cummulative bound used for early abandoning
/// w  : size of Sakoe-Chiba warpping band
/// bsf: The DTW of the current best match (used for abandoning)
/// cost_fn: Function to calculate the cost between observations
pub fn dtw<T, F>(data: &[T], query: &[T], ub: &[f64], w: usize, bsf: f64, cost_fn: &F) -> f64
where
    T: Div<Output = T> + Sub<Output = T>,
    F: Fn(&T, &T) -> f64,
{
    let mut cell_value; // TODO: REMOVE!!!

    let co;
    let li;
    if query.len() <= data.len() {
        co = query;
        li = data;
    } else {
        co = data;
        li = query;
    }

    // Instead of using matrix of size O(n^2) or O(n*w), we will reuse two array of size O(n).
    let mut prev = Array::<f64, Ix1>::from_elem(co.len() + 1, f64::INFINITY);
    let mut curr = prev.clone();
    let mut cost_tmp;

    let mut j; // Column index/index of the shorter sequence
    let mut c = f64::INFINITY; // Cost to match observation i and j with eachother

    curr[0] = 0.0;
    let mut next_start = 0;
    let mut prev_pruning_point = 1;
    let mut pruning_point = 0;
    for i in 0..li.len() {
        println!();

        // Swap the current array with the previous array
        cost_tmp = curr;
        curr = prev;
        prev = cost_tmp;

        // Missing jStop, jStart, next_pruning_point, j ,next_start

        j = next_start;
        curr[j - 1] = f64::INFINITY;

        // cost = f64::INFINTIY missing

        while j == next_start && j < pruning_point {
            c = cost_fn(&li[i], &co[j]);
            cell_value = c + f64::min(prev[j], prev[j - 1]);
            print!("{} ", cell_value);
            curr[j] = cell_value;
            if curr[j] <= ub[j] {
                next_pruning_point = j + 1;
            } else {
                next_start += 1;
            }
            j += 1;
        }
        while j < pruning_point {
            c = cost_fn(&li[i], &co[j]);
            cell_value = c + f64::min(curr[j - 1], f64::min(prev[j], prev[j - 1]));
            print!("{} ", cell_value);
            curr[j] = cell_value;
            if curr[j] <= ub[j] {
                next_pruning_point = j + 1;
            }
            j += 1;
        }
        if j <= co.len() {
            c = cost_fn(&li[i], &co[j]);
            if j == next_start {
                cell_value = c + prev[j - 1];
                print!("{} ", cell_value);
                curr[j] = cell_value;
                if curr[j] <= ub[j] {
                    pruning_point = j + 1;
                } else {
                    return f64::INFINITY;
                }
            } else {
                cell_value = c + f64::min(curr[j - 1], prev[j - 1]);
                print!("{} ", cell_value);
                curr[j] = cell_value;
                if curr[j] <= ub[j] {
                    pruning_point = j + 1;
                }
            }
            j += 1;
        } else if j == next_start {
            return f64::INFINITY;
        }
        while j == pruning_point && j <= co.len() {
            cell_value = c + curr[j - 1];
            print!("{} ", cell_value);
            curr[j] = cell_value;
            if curr[j] <= ub[j] {
                pruning_point = j + 1;
            }
            j += 1;
        }
        prev_pruning_point = pruning_point;
    }
    if prev_pruning_point <= co.len() {
        return f64::INFINITY;
    }
    curr[co.len()]
}

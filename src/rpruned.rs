use std::cmp::Ordering;

/// Calculate the Dynamic Time Wrapping distance
/// Inputs:
/// series_a, series_b: sequences of observations
/// cb : cummulative bound used for early abandoning
/// w  : size of Sakoe-Chiba warpping band
///      w MUST always be less than the length of
///      the shortest sequence, otherwise the
///      function panics
/// bsf: The DTW of the current best match (used for abandoning)
/// cost_fn: Function to calculate the cost between observations
///
/// This implemenatation was inspired by the algorithm suggested
/// in https://arxiv.org/abs/2010.05371
pub fn dtw<T, F>(
    series_a: &[T],
    series_b: &[T],
    cb: Option<&[f64]>,
    w: usize,
    bsf: f64,
    cost_fn: &F,
) -> f64
where
    F: Fn(&T, &T) -> f64,
{
    let seq_short; // The shorter of the two sequences
    let seq_long; // The longer of the two sequences

    // Find out, which of the sequences is shorter
    // This is important, because the space complexity is O(n)
    // So by choosing the shorter of the sequences, we save space
    match series_b.len().cmp(&series_a.len()) {
        Ordering::Less | Ordering::Equal => {
            seq_short = series_b;
            seq_long = series_a;
        }
        Ordering::Greater => {
            seq_short = series_a;
            seq_long = series_b;
        }
    }

    // Test if the warping window is so small, that it is
    // impossible to find a valid result
    if w < seq_long.len() - seq_short.len() {
        return f64::INFINITY;
    }

    let mut ub;
    if let Some(cb) = cb {
        if w >= cb.len() {
            panic!(
                "w is greater than the length of the 
                cumulative bound! w was {}",
                w
            );
        };
        ub = bsf - cb[w]
    } else {
        ub = bsf
    };

    let mut j; // Column index/index of the shorter sequence
    let mut c; // Cost to match observation i and j with each other

    // Instead of using matrix of size O(n^2) or O(n*w),
    // we reuse two Vecs of size O(n).
    let mut prev = vec![f64::INFINITY; seq_short.len() + 1];
    let mut curr = prev.clone();
    let mut cost_tmp;

    curr[0] = 0.0;
    let mut next_start = 0; // column to start at in the next row
    let mut prev_pruning_point = 0; // column at which the previous
                                    // row was pruned
    let mut pruning_point = 0; // column at which the current row
                               // is pruned. This gets pushed back

    let mut warp_band_begin; // column at which the warping
                             // band begins
    let mut warping_band_end; // column at which the warping
                              // band ends

    // For each row of the cost matrix
    'row_loop: for i in 0..seq_long.len() {
        // Begin at the start column
        warp_band_begin = i.saturating_sub(w);

        // If the next start is lower than the warp_band_begin,..
        if next_start < warp_band_begin {
            // and we have not reached the end of
            // the row we need to move the start
            if next_start <= prev_pruning_point {
                next_start = warp_band_begin;
            }
            // if we did reach the end of the row, we can't
            // find a start so we return infinity
            else {
                return f64::INFINITY;
            }
        }

        // Begin the calculation in the column of the next start
        j = next_start;

        // Calculate the end of the row
        warping_band_end = usize::min(i + w, seq_short.len() - 1);

        // Calculate the upper bound for early abandoning
        if i + w < seq_short.len() - 1 {
            if let Some(cb) = cb {
                ub = bsf - cb[i + w + 1]
            };
        }

        // Swap the current array with the previous array
        cost_tmp = curr;
        curr = prev;
        prev = cost_tmp;

        // The end of the row is stored in the variable pruning_point
        // Before the calculations of the columns begin, this point
        // is stored in the variable prev_pruning_point
        prev_pruning_point = pruning_point;

        // Set the value left of the start to infinity so we
        // can always use that value
        curr[j] = f64::INFINITY; // This should be uneccessary

        // While we have not found a start (a cell with a lower
        // cost than the UB)
        while j == next_start {
            match j.cmp(&prev_pruning_point) {
                // If the column is smaller than the
                // previous_pruning_point, we can have valid
                // warping paths from the top and top left of the
                // cell. Since we have not found a start yet, all
                // values left of the current cell must exceed the
                // UB so we don't bother looking at them
                Ordering::Less => {
                    c = cost_fn(&seq_long[i], &seq_short[j]);
                    curr[j + 1] = c + f64::min(prev[j + 1], prev[j]);
                }
                // If the column equals the previous_pruning_point,
                // we still have a chance to find a valid match but
                // the top left subsequence is the only possibility
                // for a valid path
                Ordering::Equal => {
                    c = cost_fn(&seq_long[i], &seq_short[j]);
                    curr[j + 1] = c + prev[j];
                }
                // If j > prev_pruning_point and we haven't found
                // a start yet, we can abandon the calculation
                Ordering::Greater => {
                    return f64::INFINITY;
                }
            }
            // If the calculated sum of costs is lower than the UB,
            // then we found a valid match, so the pruning point of
            // this row must be further to the right.
            // Also we found a valid start, so we DON'T push the
            // next_start to the right
            if curr[j + 1] < ub {
                pruning_point = j + 1;
            // If the cost is higher, it's not a valid match so we
            // need to continue looking for a start.
            // We push the next_start to the right
            } else {
                next_start += 1;
            }
            // Increase the column index
            j += 1;

            // Check if we exceeded the warping end without
            // having found a start
            if j > warping_band_end && j == next_start {
                return f64::INFINITY;
            }
        }
        // Once we found a start, we can now also have warping
        // paths coming from the left of the current cell, so
        // we must consider that cell now too.
        // While the column is lower than the
        // previous_pruning_point, we can have warping paths
        // comming from the left, top left and top cells.
        // The previous_pruning_point can never be higher
        // than co.len()+1 so we cannot exceed the cost
        // matrix in this loop
        while j < prev_pruning_point {
            c = cost_fn(&seq_long[i], &seq_short[j]);
            curr[j + 1] = c + f64::min(curr[j], f64::min(prev[j + 1], prev[j]));
            if curr[j + 1] < ub {
                pruning_point = j + 1;
            }
            j += 1;
        }
        // When reaching this point, we found a start and
        // reached the prev_pruning_point, but if we have
        // not reached the warping_end, we can still find
        // valid warping paths so we continue.
        // The only possible warping paths are the
        // left and top-left cells
        // At this point j == prev_pruning_point
        if j <= warping_band_end {
            c = cost_fn(&seq_long[i], &seq_short[j]);
            curr[j + 1] = c + f64::min(curr[j], prev[j]);

            if curr[j + 1] < ub {
                pruning_point = j + 1;
            }

            // Increase the column and check if we
            // reached the end of the row
            j += 1;
        }

        // Once we passed the prev_pruning_point but have
        // not reached the warping_end, the only possible
        // warping paths are from the left and we can
        // start with the next row once we found a value
        // that is greater than the UB
        while j <= warping_band_end {
            c = cost_fn(&seq_long[i], &seq_short[j]);
            curr[j + 1] = c + curr[j];
            if curr[j + 1] < ub {
                pruning_point = j + 1;
            } else {
                // We found a value that is greater than
                // the UB. All consecutive values must be
                // even greater, so we can start with
                // the next row
                continue 'row_loop;
            }
            j += 1;
        }
    }

    // The boundary constraint dictates, that the last
    // points must match. This only is the case, if
    // the pruning_point of the last row is pushed
    // all the way to the right
    if pruning_point == seq_short.len() {
        curr[seq_short.len()]
    } else {
        f64::INFINITY
    }
}

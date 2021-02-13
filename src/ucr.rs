/// Calculate the Dynamic Time Wrapping distance
/// data, query: data and query time series, respectively
/// cb : cummulative bound used for early abandoning
/// w  : size of Sakoe-Chiba warpping band
/// bsf: The DTW of the current best match (used for abandoning)
/// cost_fn: Function to calculate the cost between observations
pub fn dtw<T, F>(
    data: &[T],
    query: &[T],
    cb: Option<&[f64]>,
    w: usize,
    bsf: f64,
    cost_fn: &F,
) -> f64
where
    F: Fn(&T, &T) -> f64,
{
    let data_len = data.len(); // Also called n

    // We reuse two Vecs so we have a space complexity of O(n)
    let mut cost = vec![f64::INFINITY; data_len];
    let mut cost_prev = cost.clone();
    let mut cost_tmp;

    // Variables to implement the pruning - PrunedDTW

    let mut j_init;
    let mut j_end;

    let mut start_found: bool; // Flag to remember, that a start for the next row was found
    let mut next_start = 0; // Index at which the calculation will start for the next row
    let mut ec = 0; // Index of the last column in which the cost was lower than UB
    let mut ec_prev = 0; // Index at which we ended the previous row

    let mut ub = if let Some(cb) = cb { bsf - cb[w] } else { bsf };

    //let mut ub = bsf - cb[w]; // Upper bound which the cost needs to stay below, otherwise we know we can not find a cost that is lower than the bsf

    let (mut cell_top, mut cell_left, mut cell_top_left, mut min_cost);

    for i in 0..data_len {
        min_cost = f64::MAX;

        start_found = false;

        // Calculate the first and last viable columns where we could find valid matches for the warping path
        j_init = usize::max(i.saturating_sub(w), next_start);
        j_end = usize::min(i + w, query.len() - 1);

        // For each value that can be potentially matched from the query, do
        for j in j_init..j_end + 1 {
            // Initialize the values for the first point of the sequences
            if (i == 0) && (j == 0) {
                cost[j] = cost_fn(&data[0], &query[0]);
                min_cost = cost[j];
                start_found = true;
                continue;
            }

            // Now we calculate the values of the neighboring cells from which a warping path can connect to
            if j == j_init {
                cell_left = f64::INFINITY;
            } else {
                cell_left = cost[j - 1];
            }
            if (i == 0) || (j == i + w) || (j > ec_prev) {
                cell_top = f64::INFINITY;
            } else {
                cell_top = cost_prev[j];
            }
            if (i == 0) || (j == 0) || (j > ec_prev + 1) {
                cell_top_left = f64::INFINITY;
            } else {
                cell_top_left = cost_prev[j - 1];
            }

            // Classic DTW calculation
            cost[j] = f64::min(f64::min(cell_top, cell_top_left), cell_left)
                + cost_fn(&data[i], &query[j]);

            // Find the minimum cost of the row
            if cost[j] < min_cost {
                min_cost = cost[j];
            }

            // If the cost is lower than the UB,
            if cost[j] < ub {
                // and we previously have not found a start, we now found one and store its index
                if !start_found {
                    next_start = j;
                    start_found = true;
                }
                // and we remember the column at which we found a cost that is lower than the ub
                ec = j;
            // If the cost was greater or equal to the UB and we are past the end column of the previous row, we can abandon the rest of the row
            } else if j > ec_prev + 1 {
                // We break the loop, because we can contine with the next row
                break;
            }
        }

        if i + w < data_len - 1 {
            ub = if let Some(cb) = cb {
                bsf - cb[i + w + 1]
            } else {
                bsf
            };
        }

        // We can abandon early if the minimum cost is larger than the UB
        if min_cost >= ub {
            return f64::INFINITY;
        }

        // Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;

        if next_start > 0 {
            cost_prev[next_start - 1] = f64::INFINITY;
        }

        ec_prev = ec;
    }
    // If pruned in the last row, the ec did not reach the end. In that case we did not beat the bsf and return INFINITY
    if ec < query.len() - 1 {
        return f64::INFINITY;
    }

    cost_prev[data_len - 1]
}

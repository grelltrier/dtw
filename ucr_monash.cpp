double dtw(
    const double *lines,
    const double *cols,
    const double *cb,
    int l,
    int w,
    double bsf
    )
{
  // 1) --- Create the upper bound from bsf and a margin
  double UB= bsf - cb[w+1];

  // 2) --- Alias in line/column concept: we only allocate for the columns, using the smallest possible dimension.
  const size_t nbcols = l;
  const size_t nblines = l;

  // 3) --- Cap the windows.
  if (w > nblines) { w = nblines; }

  // 4) --- Buffers allocations
  // Add an extra column for the "matrix border" condition, init to +INF.
  // Using an unique contiguous array. Base indices are:
  // 'c' for the current row,
  // 'p' for the previous one
  std::vector<double> buffers_v((1+nbcols) * 2, POSITIVE_INFINITY);
  double *buffers = buffers_v.data();
  size_t c{1}, p{nbcols+2};                 // Account for the extra column (in front)

  // 5) --- Computation of DTW
  buffers[c-1] = 0;
  size_t next_start{0};
  size_t pruning_point{0};

  for(size_t i=0; i<nblines; ++i) {
    // --- --- --- --- Swap and variables init
    std::swap(c, p);
    const double li = lines[i];
    const std::size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
    const std::size_t jStart = std::max(cap_start_index_to_window(i, w), next_start);
    std::size_t next_pruning_point = jStart; // Next pruning point init at the start of the line
    std::size_t j = jStart;
    next_start = jStart;
    // --- --- --- --- Init the first column
    buffers[c+j-1] = POSITIVE_INFINITY;
    double cost = POSITIVE_INFINITY;
    // --- --- --- --- Compute DTW up to the pruning point while advancing next_start: diag and top
    for(; j==next_start && j < pruning_point; ++j) {
      const auto d = square_dist(li, cols[j]);
      cost = std::min(buffers[p + j - 1], buffers[p + j]) + d;
      buffers[c + j] = cost;
      if(cost<=UB){ next_pruning_point = j + 1;} else { ++next_start; }
    }
    // --- --- --- --- Compute DTW up to the pruning point without advancing next_start: prev, diag, top
    for(; j < pruning_point; ++j) {
      const auto d = square_dist(li, cols[j]);
      cost = std::min(cost, std::min(buffers[p + j - 1], buffers[p + j])) + d;
      buffers[c + j] = cost;
      if(cost<=UB){ next_pruning_point = j + 1;}
    }
    // --- --- --- --- Compute DTW at "pruning_point": 2 cases
    if(j<jStop){
      const auto d = square_dist(li, cols[j]);
      if(j==next_start){ // Advancing next start: only diag. Done if v>UB.
        cost = buffers[p + j - 1] + d;
        buffers[c + j] = cost;
        if(cost<=UB){ next_pruning_point = j + 1;} else {return POSITIVE_INFINITY; }
      } else { // Not advancing next start: at least a path possible in previous cells.
        cost = std::min(cost, buffers[p + j - 1]) + d;
        buffers[c + j] = cost;
        if(cost<=UB){ next_pruning_point = j + 1;}
      }
      ++j;
    } else if(j==next_start) { return POSITIVE_INFINITY; }
    // --- --- --- --- Compute DTW after "pruning_point": prev. Go on while we advance the next pruning point.
    for(;j==next_pruning_point && j<jStop;++j){
      const auto d = square_dist(li, cols[j]);
      cost = cost + d;
      buffers[c + j] = cost;
      if(cost<=UB){ ++next_pruning_point; }
    }
    // --- --- --- --- Row done, update the pruning point variable and the upper bound
    pruning_point=next_pruning_point;
    if(i+w < nbcols){ UB = bsf - cb[i+w+1]; }
  }// End for i loop


  // 6) --- If the pruning_point did not reach the number of columns, we pruned something
  if(pruning_point != nbcols){ return POSITIVE_INFINITY; } else {
    return buffers[c+nbcols-1];
  }
}
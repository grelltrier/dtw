use core::ops::AddAssign;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_stats::DeviationExt;
use num_traits::{cast::ToPrimitive, sign::Signed};

pub mod costs;
pub mod naive;
#[cfg(test)]
mod tests;
pub mod ucr;

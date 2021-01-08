struct Envelope {
    upper: Vec<f64>,
    lower: Vec<f64>,
    calculated_till: usize,
}

/*
        let len = q.len();
        //let bsf = bsf.unwrap_or(f64::INFINITY);

        let mut d;
        let mut lb;

        // 1 point at front and back
        let x0 = (t[j] - mean) / std;
        let y0 = (t[(j + len - 1)] - mean) / std;
        lb = dist(x0, q[0]) + dist(y0, q[len - 1]);
        if lb >= bsf {
            return (lb, 1);
        }

        // 2 points at front
        let x1 = (t[(j + 1)] - mean) / std;
        d = f64::min(dist(x1, q[0]), dist(x0, q[1]));
        d = f64::min(d, dist(x1, q[1]));
        lb += d;
        if d >= bsf {
            return (lb, 2);
        }
        if lb >= bsf {
            return (lb, 1);
        }

        // 2 points at back
        let y1 = (t[(j + len - 2)] - mean) / std;
        d = f64::min(dist(y1, q[len - 1]), dist(y0, q[len - 2]));
        d = f64::min(d, dist(y1, q[len - 2]));
        lb += d;
        if d >= bsf {
            return (lb, len - 2 + 1);
        }
        if lb >= bsf {
            return (lb, 1);
        }

        // 3 points at front
        let x2 = (t[(j + 2)] - mean) / std;
        d = f64::min(dist(x0, q[2]), dist(x1, q[2]));
        d = f64::min(d, dist(x2, q[2]));
        d = f64::min(d, dist(x2, q[1]));
        d = f64::min(d, dist(x2, q[0]));
        lb += d;
        if d >= bsf {
            return (lb, 3);
        }
        if lb >= bsf {
            return (lb, 1);
        }
        // 3 points at back
        let y2 = (t[(j + len - 3)] - mean) / std;
        d = f64::min(dist(y0, q[len - 3]), dist(y1, q[len - 3]));
        d = f64::min(d, dist(y2, q[len - 3]));
        d = f64::min(d, dist(y2, q[len - 2]));
        d = f64::min(d, dist(y2, q[len - 1]));
        lb += d;
        if d >= bsf {
            return (lb, len - 3 + 1);
        }

        (lb, 1)
*/

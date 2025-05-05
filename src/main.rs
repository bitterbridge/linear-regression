use rand::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Point(pub f64, pub f64);

#[derive(Clone, Copy)]
pub struct Line(pub f64, pub f64);

impl Line {
    fn predicted_y(&self, x: f64) -> f64 {
        self.0 * x + self.1
    }
}

impl std::fmt::Debug for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Line({:.4}, {:.4})", self.0, self.1)
    }
}

pub fn build_vector<R: Rng>(target_line: &Line, length: usize, _rng: &mut R) -> Vec<Point> {
    let mut result = Vec::with_capacity(length);
    let min_x: f64 = -1.0;
    let max_x: f64 = 1.0;
    let range_x = max_x - min_x;
    let point_count = 1000;
    let dx = range_x / point_count as f64;
    for i in 0..length {
        let x: f64 = dx * i as f64 + min_x;
        let y: f64 = target_line.0 * x + target_line.1;
        result.push(Point(x, y));
    }
    result
}

pub fn simple_trick<R: Rng>(line: &mut Line, point: &Point, rng: &mut R) {
    let random_m: f64 = rng.random_range(0.00001..0.00009);
    let random_b: f64 = rng.random_range(0.00001..0.00009);
    if line.predicted_y(point.0) > point.1 {
        line.0 -= random_m;
        line.1 -= random_b;
    } else {
        line.0 += random_m;
        line.1 += random_b;
    }
}

pub fn train_simple_trick<R: Rng>(
    mut line: Line,
    points: &Vec<Point>,
    iterations: usize,
    target: &Line,
    rng: &mut R,
) -> Line {
    println!("Simple trick ({} iterations)", iterations);
    println!("Targeting {:?}", target);
    let reporting_interval = iterations / 20;
    for i in 0..iterations {
        if let Some(random_point) = points.choose(rng) {
            simple_trick(&mut line, &random_point, rng);
        }
        if i % reporting_interval == 0 {
            println!("Iteration {}: {:?}", i, line);
        }
    }
    println!("Final {:?} (should be {:?})", line, target);
    line
}

pub fn square_trick(line: &mut Line, point: &Point, learning_rate: f64) {
    let y_diff = point.1 - line.predicted_y(point.0);
    line.0 += learning_rate * point.0 * y_diff;
    line.1 += learning_rate * y_diff;
}

pub fn train_square_trick<R: Rng>(
    mut line: Line,
    points: &Vec<Point>,
    iterations: usize,
    target: &Line,
    learning_rate: f64,
    rng: &mut R,
) -> Line {
    println!("Square trick ({} iterations)", iterations);
    println!("Targeting {:?}", target);
    let reporting_interval = iterations / 50;
    for i in 0..iterations {
        if let Some(random_point) = points.choose(rng) {
            square_trick(&mut line, &random_point, learning_rate);
        }
        if i % reporting_interval == 0 {
            println!("Iteration {}: {:?}", i, line);
        }
    }
    println!("Final {:?} (should be {:?})", line, target);
    line
}

fn main() {
    let actual_m = 1.0;
    let actual_b = 0.0;
    let initial_m = 10.0;
    let initial_b = 10.0;
    let length = 1000;
    let mut rng = rand::rng();
    let target_line = Line(actual_m, actual_b);
    let points = build_vector(&target_line, length, &mut rng);

    train_simple_trick(
        Line(initial_m, initial_b),
        &points,
        30000,
        &target_line,
        &mut rng,
    );

    train_square_trick(
        Line(initial_m, initial_b),
        &points,
        30000,
        &target_line,
        0.001,
        &mut rng,
    );
}

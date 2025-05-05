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

pub fn build_vector<R: Rng>(
    target_line: &Line,
    length: usize,
    variation: f64,
    rng: &mut R,
) -> Vec<Point> {
    let mut result = Vec::with_capacity(length);
    let min_x: f64 = -1.0;
    let max_x: f64 = 1.0;
    let range_x = max_x - min_x;
    let point_count = 1000;
    let dx = range_x / point_count as f64;
    for i in 0..length {
        let x: f64 = dx * i as f64 + min_x;
        let offset = rng.random_range(-variation..variation);
        let y: f64 = target_line.0 * x + target_line.1 + offset;
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
    early_converge: f64,
    rng: &mut R,
) -> Line {
    println!("==========================================");
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
        if (line.0 - target.0).abs() < early_converge && (line.1 - target.1).abs() < early_converge
        {
            println!("Converged early on iteration {}!", i);
            break;
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
    early_converge: f64,
    rng: &mut R,
) -> Line {
    println!("==========================================");
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
        if (line.0 - target.0).abs() < early_converge && (line.1 - target.1).abs() < early_converge
        {
            println!("Converged early on iteration {}!", i);
            break;
        }
    }
    println!("Final {:?} (should be {:?})", line, target);
    line
}

pub fn absolute_trick(line: &mut Line, point: &Point, learning_rate: f64) {
    let y_diff = point.1 - line.predicted_y(point.0);
    if y_diff > 0.0 {
        line.0 += learning_rate * point.0;
        line.1 += learning_rate;
    } else {
        line.0 -= learning_rate * point.0;
        line.1 -= learning_rate;
    }
}

pub fn train_absolute_trick<R: Rng>(
    mut line: Line,
    points: &Vec<Point>,
    iterations: usize,
    target: &Line,
    learning_rate: f64,
    early_converge: f64,
    rng: &mut R,
) -> Line {
    println!("==========================================");
    println!("Absolute trick ({} iterations)", iterations);
    println!("Targeting {:?}", target);
    let reporting_interval = iterations / 50;
    for i in 0..iterations {
        if let Some(random_point) = points.choose(rng) {
            absolute_trick(&mut line, &random_point, learning_rate);
        }
        if i % reporting_interval == 0 {
            println!("Iteration {}: {:?}", i, line);
        }
        if (line.0 - target.0).abs() < early_converge && (line.1 - target.1).abs() < early_converge
        {
            println!("Converged early on iteration {}!", i);
            break;
        }
    }
    println!("Final {:?} (should be {:?})", line, target);
    line
}

fn main() {
    let mut rng = rand::rng();
    let actual_m = rng.random_range(-100.0..100.0);
    let actual_b = rng.random_range(-100.0..100.0);
    let initial_m = rng.random_range(-100.0..100.0);
    let initial_b = rng.random_range(-100.0..100.0);
    let length = 1000;
    let target_line = Line(actual_m, actual_b);
    let variation = 0.00001;
    let early_converge = 0.001;
    let points = build_vector(&target_line, length, variation, &mut rng);

    train_simple_trick(
        Line(initial_m, initial_b),
        &points,
        5000000,
        &target_line,
        early_converge,
        &mut rng,
    );

    train_square_trick(
        Line(initial_m, initial_b),
        &points,
        5000000,
        &target_line,
        0.001,
        early_converge,
        &mut rng,
    );

    train_absolute_trick(
        Line(initial_m, initial_b),
        &points,
        5000000,
        &target_line,
        0.001,
        early_converge,
        &mut rng,
    );
}

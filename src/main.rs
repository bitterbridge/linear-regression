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
    let dx = range_x / length as f64;
    for i in 0..length {
        let x: f64 = dx * i as f64 + min_x;
        let offset = rng.random_range(-variation..variation);
        let y: f64 = target_line.0 * x + target_line.1 + offset;
        result.push(Point(x, y));
    }
    result
}

pub fn train_model<F: FnMut(&mut Line, &Point)>(
    name: &str,
    mut line: Line,
    points: &[Point],
    epochs: usize,
    target: &Line,
    test_points: &[Point],
    early_converge: f64,
    mut update_fn: F,
) -> Line {
    println!("\n");
    println!("==========================================");
    println!("{} ({} epochs)", name, epochs);
    println!("Starting {:?}", line);
    println!("Targeting {:?}", target);
    let reporting_interval = epochs / 10;
    for i in 0..epochs {
        for point in points {
            update_fn(&mut line, point);
        }
        if i % reporting_interval == 0 {
            println!("Epoch {}: {:?}", i, line);
        }
        if (line.0 - target.0).abs() < early_converge && (line.1 - target.1).abs() < early_converge
        {
            println!("Converged at epoch {}", i);
            break;
        }
    }
    println!("Final {:?} (should be {:?})", line, target);
    test_line(name, &line, test_points);
    line
}

pub fn test_line(name: &str, line: &Line, points: &[Point]) {
    let mut absolute_error = 0.0;
    let mut square_error = 0.0;
    for point in points {
        let dy = (line.predicted_y(point.0) - point.1).abs();
        absolute_error += dy;
        square_error += dy.powi(2);
    }
    println!("{} Absolute Error: {}", name, absolute_error);
    println!("{} Square Error: {}", name, square_error);
    println!(
        "{} Mean Absolute Error: {}",
        name,
        absolute_error / points.len() as f64
    );
    println!(
        "{} Mean Square Error: {}",
        name,
        square_error / points.len() as f64
    );
    println!(
        "{} Root Mean Square Error: {}",
        name,
        (square_error / points.len() as f64).sqrt()
    );
}

fn main() {
    let mut rng = rand::rng();
    let actual_m = rng.random_range(-100.0..100.0);
    let actual_b = rng.random_range(-100.0..100.0);
    let initial_m = rng.random_range(-100.0..100.0);
    let initial_b = rng.random_range(-100.0..100.0);
    let epochs = 10000;
    let length = 1000;
    let target_line = Line(actual_m, actual_b);
    let variation = 0.05;
    let early_converge = 0.001;
    let learning_rate = 0.001;
    let test_length = 100;
    let points = build_vector(&target_line, length, variation, &mut rng);
    let test_points = build_vector(&target_line, test_length, variation, &mut rng);

    train_model(
        "Simple Trick",
        Line(initial_m, initial_b),
        &points,
        epochs,
        &target_line,
        &test_points,
        early_converge,
        |line, point| {
            if line.predicted_y(point.0) > point.1 {
                line.0 -= learning_rate;
                line.1 -= learning_rate;
            } else {
                line.0 += learning_rate;
                line.1 += learning_rate;
            }
        },
    );

    train_model(
        "Square Trick",
        Line(initial_m, initial_b),
        &points,
        epochs,
        &target_line,
        &test_points,
        early_converge,
        |line, point| {
            let y_diff = point.1 - line.predicted_y(point.0);
            line.0 += learning_rate * point.0 * y_diff;
            line.1 += learning_rate * y_diff;
        },
    );

    train_model(
        "Absolute Trick",
        Line(initial_m, initial_b),
        &points,
        epochs,
        &target_line,
        &test_points,
        early_converge,
        |line, point| {
            let y_diff = point.1 - line.predicted_y(point.0);
            if y_diff > 0.0 {
                line.0 += learning_rate * point.0;
                line.1 += learning_rate;
            } else {
                line.0 -= learning_rate * point.0;
                line.1 -= learning_rate;
            }
        },
    );
}

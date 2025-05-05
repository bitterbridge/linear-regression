# linear-regression

Simple linear regression implementation in Rust.

This project implements a few fundamental linear regression algorithms:
- the "simple" trick
- the "square" trick
- the "absolute" trick

Linear regression algorithms work as follows:
- given a data set of points
- pick a random point in the data set
- given the point's _x_, predict its _y_
- move the line slightly closer to the actual _y_

Determining _how_ to move the line closer to one that trends ideally through the dataset is how these algorithms differ:
- the "simple trick": moving the line up or down and increasing or decreasing the slope by a _random_ small amount
- the "square trick":
  - moving the line up or down by a small amount proportional to the product of the discrepancy and the target value
  - increasing or decreasing the slope by a small amount proportional to the discrepancy
- the "absolute trick":
  - moving the line up or down by a small amount proportional to the target value
  - increasing or decreasing the slope by a small amount proportional to the discrepancy

## Instructions

To build, all you really need do is check out this repository and run it using [Cargo](https://doc.rust-lang.org/cargo/).

```bash
cargo run
```

This will pick a line with random slope and y-intercept, distribute points throughout, and then perform a linear regression using each of the algorithms listed above, concluding by showing the results and how they match against the actual line.

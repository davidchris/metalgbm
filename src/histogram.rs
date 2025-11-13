pub struct Histogram {
    bins: Vec<f32>,
    gradients: Vec<f32>,
    hessians: Vec<f32>, // first derivative of loss function
}

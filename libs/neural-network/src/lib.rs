use rand::{Rng, RngCore};

pub struct Network {
    layers: Vec<Layer>,
}

pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.propagate(inputs);
        }

        inputs
    }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| {
                Layer::random(rng, layers[0].neurons, layers[1].neurons)
            })
            .collect();

        Self { layers }
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            let output = neuron.propagate(&inputs);
            outputs.push(output);
        }

        outputs
    }

    pub fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();
    
        Self { neurons }
    }
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let mut output = 0.0;

        for i in 0..inputs.len() {
            output += inputs[i] * self.weights[i];
        }
        //.max implements relu activation function for each neuron
        (self.bias + output).max(0.0)
    }

    pub fn random(
        rng: &mut dyn rand::RngCore,
        output_size: usize,
    ) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::seed_from_u64(2);
            let neuron = Neuron::random(&mut rng, 4);
        
            approx::assert_relative_eq!(neuron.bias, -0.6129729);
        
            approx::assert_relative_eq!(neuron.weights.as_slice(), [
                0.7626815, 
                0.044567704, 
                0.09686196, 
                -0.93544626]
                .as_ref());
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };
        
            // Ensures relu works
            approx::assert_relative_eq!(
                neuron.propagate(&[-10.0, -10.0]),
                0.0,
            );
        
            // `0.5` and `1.0` chosen by a fair dice roll:
            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
            );
        }
    }
}
use blas::Blas;
use std::marker::PhantomData;

type Error = String;

pub trait Activator<F> {
    fn activate(x: F) -> F;
}

pub struct NeuralNet<F, A: Activator<F>> {
    phantom: PhantomData<A>,
    inputs: usize,
    input_neurons: usize,
    folding_step: isize,
    layers: usize,
    coefs: Vec<F>
}

pub struct Context<F, B: Blas<F>> {
	blas: B,
	vec1: Vec<F>,
	vec2: Vec<F>
}

impl<F, B: Blas<F>> Context<F, B> {
	pub fn compute<'a, A: Activator<F>>(&'a mut self, net: &NeuralNet<F, A>,
		                                input: &[F]) -> Result<&'a [F], Error> {

		Ok(&self.vec1[..])
	}
}

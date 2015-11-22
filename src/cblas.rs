use blas::{Blas, Error, Transpose};
use dylib::DynamicLibrary;
use std::mem::transmute;
use std::path::Path;

pub struct Cblas {
    #[allow(dead_code)]
    dylib: DynamicLibrary,
    sgemv: usize,
    dgemv: usize
}

const ROW_MAJOR: i32 = 101;

fn trans_val(trans: Transpose) -> i32 {
    match trans {
        Transpose::No => 111,
        Transpose::Yes => 112,
        Transpose::Conjugate => 113
    }
}

impl Cblas {
    pub fn init<P: AsRef<Path>>(dlib_path: P) -> Result<Cblas, Error> {
        let dylib = try!(DynamicLibrary::open(Some(dlib_path.as_ref())));

        unsafe {
            Ok(Cblas {
                sgemv: transmute(try!(dylib.symbol::<u8>("cblas_sgemv"))),
                dgemv: transmute(try!(dylib.symbol::<u8>("cblas_dgemv"))),
                dylib: dylib,
            })
        }
    }

    #[inline]
    fn do_gemv<T>(&self, func: usize,
                  trans: Transpose, cols: usize, rows: usize, matrix: &[T],
                  in_vector: &[T], in_vector_inc: usize, product_factor: T,
                  out_vector: &mut [T], out_vector_inc: usize,
                  out_vector_factor: T) -> Result<(), Error> {
        use blas::assert_gemv;
        assert_gemv(trans, cols, rows, matrix,
            in_vector, in_vector_inc, out_vector, out_vector_inc);

        unsafe {
            let gemv: extern fn(i32, i32, i32, i32,
                                T, *const T, i32,
                                *const T, i32, T,
                                *mut T, i32)
                = transmute(func);

            gemv(ROW_MAJOR, trans_val(trans), rows as i32, cols as i32,
                product_factor, matrix.as_ptr(), (matrix.len() / rows) as i32,
                in_vector.as_ptr(), in_vector_inc as i32, out_vector_factor,
                out_vector.as_mut_ptr(), out_vector_inc as i32);
            Ok(())
        }
    }
}

impl Blas<f32> for Cblas {
    fn gemv(&mut self, trans: Transpose, cols: usize, rows: usize,
            matrix: &[f32], in_vector: &[f32], in_vector_inc: usize,
            product_factor: f32, out_vector: &mut [f32], out_vector_inc: usize,
            out_vector_factor: f32,) -> Result<(), Error> {
        self.do_gemv(self.sgemv, trans, cols, rows, matrix,
            in_vector, in_vector_inc, product_factor,
            out_vector, out_vector_inc, out_vector_factor)
    }
}

impl Blas<f64> for Cblas {
    fn gemv(&mut self, trans: Transpose, cols: usize, rows: usize,
            matrix: &[f64], in_vector: &[f64], in_vector_inc: usize,
            product_factor: f64, out_vector: &mut [f64], out_vector_inc: usize,
            out_vector_factor: f64) -> Result<(), Error> {
        self.do_gemv(self.dgemv, trans, cols, rows, matrix,
            in_vector, in_vector_inc, product_factor,
            out_vector, out_vector_inc, out_vector_factor)
    }
}

#[cfg(test)]
pub mod tests {
    use blas::tests::*;
    use config;
    use super::*;

    #[test]
    pub fn test_gemv() {
        let config = config::Config::new().unwrap(); 
        let mut blas = Cblas::init(config.cblas_dylib_path).unwrap();
        test_qemv::<f32, Cblas>(&mut blas);
        test_qemv::<f64, Cblas>(&mut blas);
    }
}

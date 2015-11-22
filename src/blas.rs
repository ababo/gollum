#[derive(Clone, Copy)]
pub enum Transpose {
    No,
    Yes,
    Conjugate
}

pub type Error = String;

pub trait Blas<T> {
    fn gemv(&mut self, trans: Transpose, cols: usize, rows: usize, matrix: &[T],
            in_vector: &[T], in_vector_inc: usize, product_factor: T,
            out_vector: &mut [T], out_vector_inc: usize, out_vector_factor: T)
        -> Result<(), Error>;
}

#[inline]
pub fn assert_matrix<T>(cols: usize, rows: usize, matrix: &[T]) {
    assert!(matrix.len() / rows >= cols);
}

#[inline]
pub fn assert_vector<T>(vector: &[T], vector_inc: usize) {
    assert!(vector.len() % vector_inc == 0);
}

#[inline]
pub fn assert_gemv<T>(trans: Transpose, cols: usize, rows: usize, matrix: &[T],
                      in_vector: &[T], in_vector_inc: usize,
                      out_vector: &[T], out_vector_inc: usize) {
    assert_matrix(cols, rows, matrix);
    assert_vector(in_vector, in_vector_inc);
    assert_vector(out_vector, out_vector_inc);
    match trans {
        Transpose::No => {
            assert!(in_vector.len() / in_vector_inc == cols);
            assert!(out_vector.len() / out_vector_inc == rows);
        },
        _ => {
            assert!(in_vector.len() / in_vector_inc == rows);
            assert!(out_vector.len() / out_vector_inc == cols);
        }
    };
}

#[cfg(test)]
pub mod tests {
    use num::traits::{NumCast, ToPrimitive};
    use super::*;

    #[test]
    #[should_panic]
    pub fn test_assert_matrix() {
        assert_matrix(2, 3, &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    pub fn test_assert_vector() {
        assert_vector(&[1, 2, 3], 2);
    }

    #[test]
    #[should_panic]
    pub fn test_assert_gemv() {
        assert_gemv(Transpose::No, 3, 2, &[1, 2, 3, 4, 5, 6],
            &[1, 2], 1, &[1, 2, 3], 1);
    }

    #[inline]
    pub fn cast_val<F: ToPrimitive+Copy, T: NumCast>(val: F) -> T {
        T::from(val).unwrap()
    }

    #[inline]
    pub fn cast_slice<F: ToPrimitive+Copy,
                         T: NumCast>(slice: &[F]) -> Vec<T> {
        slice.iter().map(|a: &F| cast_val(*a)).collect()
    }

    pub fn test_qemv<F: NumCast+Copy, B: Blas<F>>(blas: &mut B) {
        let m: Vec<F> = cast_slice(&vec![1, 2, 3, 4, 5, 6, 7, 8][..]);
        let i: Vec<F> = cast_slice(&vec![9, 10, 11, 12, 13, 14][..]);

        let mut o: Vec<F> = cast_slice(&vec![15, 16, 17, 18, 19, 20][..]);
        blas.gemv(Transpose::No, 3, 2, &m[..],
            &i[..], 2, cast_val(2), &mut o[..], 3, cast_val(3)).unwrap();
        assert_eq!(&[185, 16, 17, 458, 19, 20][..], &cast_slice(&o[..])[..]);

        o = cast_slice(&vec![15, 16, 17, 18, 19, 20][..]);
        blas.gemv(Transpose::Yes, 3, 2, &m[..],
            &i[..], 3, cast_val(2), &mut o[..], 2, cast_val(3)).unwrap();
        assert_eq!(&[183, 16, 231, 18, 279, 20][..], &cast_slice(&o[..])[..]);

        o = cast_slice(&vec![15, 16, 17, 18, 19, 20][..]);
        blas.gemv(Transpose::Conjugate, 3, 2, &m[..],
            &i[..], 3, cast_val(2), &mut o[..], 2, cast_val(3)).unwrap();
        assert_eq!(&[183, 16, 231, 18, 279, 20][..], &cast_slice(&o[..])[..]);
    }
}

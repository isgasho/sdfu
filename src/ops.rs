//! Operations you can perform to combine two SDFs.
use super::*;

macro_rules! impl_ops {
    ($($t:ty),+) => {
        /// A function which can get the minimum between two SDFs.
        /// This is useful because sometimes we want to be able to
        /// interpolate between the minimums for 'soft blending' between
        /// two SDFs.
        pub trait MinFunction<T>: Copy {
            fn min(&self, a: T, b: T) -> T;
        }

        /// A function which can get the maximum between two SDFs.
        /// This is useful because sometimes we want to be able to
        /// interpolate between the maxima for 'soft blending' between
        /// two SDFs.
        pub trait MaxFunction<T>: Copy {
            fn max(&self, a: T, b: T) -> T;
        }

        /// Takes the absolute minimum of two values
        /// and returns them directly. A standard min function.
        #[derive(Clone, Copy, Debug)]
        pub struct HardBlend {}

        impl Default for HardBlend {
            fn default() -> Self {
                HardBlend {}
            }
        }

        $(impl MinFunction<$t> for HardBlend {
            fn min(&self, a: $t, b: $t) -> $t {
                a.min(b)
            }
        }

        impl MaxFunction<$t> for HardBlend {
            fn max(&self, a: $t, b: $t) -> $t {
                a.max(b)
            }
        })+

        /// Blends between two functions, smoothing between them
        /// when they are close.
        ///
        /// This uses an exponential function to smooth between the two
        /// values, and `k` controls the radius/distance of the
        /// smoothing. 32 is a good default value for `k` for this
        /// smoothing function.
        #[derive(Clone, Copy, Debug)]
        pub struct ExponentialSmoothBlend<T> {
            pub k: T,
        }

        $(impl Default for ExponentialSmoothBlend<$t> {
            fn default() -> Self {
                ExponentialSmoothBlend { k: 32.0 }
            }
        }

        impl ExponentialSmoothBlend<$t>
        {
            fn evaluate(&self, a: $t, b: $t) -> $t {
                let res = (-self.k * a).exp2() + (-self.k * b).exp2();
                -res.log2() / self.k
            }
        }

        impl MinFunction<$t> for ExponentialSmoothBlend<$t>
        {
            fn min(&self, a: $t, b: $t) -> $t {
                self.evaluate(a, b)
            }
        }

        impl MaxFunction<$t> for ExponentialSmoothBlend<$t>
        {
            fn max(&self, a: $t, b: $t) -> $t {
                -self.evaluate(-a, -b)
            }
        })+

        /// Blends between two functions, smoothing between them
        /// when they are close.
        ///
        /// This uses a polynomial function to smooth between the two
        /// values, and `k` controls the radius/distance of the
        /// smoothing. 0.1 is a good default value for `k` for this
        /// smoothign function.
        #[derive(Clone, Copy, Debug)]
        pub struct PolySmoothBlend<T> {
            pub k: T,
        }

        $(impl PolySmoothBlend<$t> {
            pub fn new(k: $t) -> Self {
                PolySmoothBlend { k }
            }
        }

        impl Default for PolySmoothBlend<$t> {
            fn default() -> Self {
                PolySmoothBlend { k: 0.1 }
            }
        }

        impl PolySmoothBlend<$t>
        {
            fn evaluate(&self, a: $t, b: $t) -> $t {
                let t = 0.5 + 0.6 * (b - a) / self.k;
                let h = Clamp::clamp(&t, 0.0, 1.0);
                b.lerp(a, h) - self.k * h * (1.0 - h)
            }
        }

        impl MinFunction<$t> for PolySmoothBlend<$t>
        {
            fn min(&self, a: $t, b: $t) -> $t {
                self.evaluate(a, b)
            }
        }

        impl MaxFunction<$t> for PolySmoothBlend<$t>
        {
            fn max(&self, a: $t, b: $t) -> $t {
                -self.evaluate(-a, -b)
            }
        })+

        /// The union of two SDFs.
        #[derive(Clone, Copy, Debug)]
        pub struct Union<S1, S2, M> {
            pub sdf1: S1,
            pub sdf2: S2,
            pub min_func: M,
        }

        impl<S1, S2> Union<S1, S2, HardBlend> {
            pub fn hard(sdf1: S1, sdf2: S2) -> Self {
                Union {
                    sdf1,
                    sdf2,
                    min_func: HardBlend::default(),
                }
            }
        }

        impl<S1, S2, M> Union<S1, S2, M>
        {
            pub fn new(sdf1: S1, sdf2: S2, min_func: M) -> Self {
                Union {
                    sdf1,
                    sdf2,
                    min_func,
                }
            }
        }

        $(impl<S1, S2> Union<S1, S2, PolySmoothBlend<$t>> {
            pub fn smooth(sdf1: S1, sdf2: S2, smoothness: $t) -> Self {
                Union {
                    sdf1,
                    sdf2,
                    min_func: PolySmoothBlend::<$t>::new(smoothness),
                }
            }
        }

        impl<V, S1, S2, M> Sdf<$t, V> for Union<S1, S2, M>
        where
            V: Vec<$t>,
            S1: Sdf<$t, V>,
            S2: Sdf<$t, V>,
            M: MinFunction<$t> + Copy,
        {
            fn dist(&self, p: V) -> $t {
                self.min_func.min(self.sdf1.dist(p), self.sdf2.dist(p))
            }
        })+

        /// Get the subtracion of two SDFs. Note that this operation is *not* commutative,
        /// i.e. `Subtraction::new(a, b) =/= Subtraction::new(b, a)`.
        #[derive(Clone, Copy, Debug)]
        pub struct Subtraction<S1, S2, M> {
            pub sdf1: S1,
            pub sdf2: S2,
            pub max_func: M,
        }

        impl<S1, S2, M> Subtraction<S1, S2, M> {
            /// Get the subtracion of two SDFs. Note that this operation is *not* commutative,
            /// i.e. `Subtraction::new(a, b) =/= Subtraction::new(b, a)`.
            pub fn new(sdf1: S1, sdf2: S2, max_func: M) -> Self {
                Subtraction {
                    sdf1,
                    sdf2,
                    max_func,
                }
            }
        }

        impl<S1, S2> Subtraction<S1, S2, HardBlend> {
            pub fn hard(sdf1: S1, sdf2: S2) -> Self {
                Subtraction {
                    sdf1,
                    sdf2,
                    max_func: HardBlend::default(),
                }
            }
        }

        $(impl<S1, S2> Subtraction<S1, S2, PolySmoothBlend<$t>> {
            pub fn smooth(sdf1: S1, sdf2: S2, smoothness: $t) -> Self {
                Subtraction {
                    sdf1,
                    sdf2,
                    max_func: PolySmoothBlend::<$t>::new(smoothness),
                }
            }
        }

        impl<V, S1, S2, M> Sdf<$t, V> for Subtraction<S1, S2, M>
        where
            V: Vec<$t>,
            S1: Sdf<$t, V>,
            S2: Sdf<$t, V>,
            M: MaxFunction<$t>,
        {
            fn dist(&self, p: V) -> $t {
                self.max_func.max(-self.sdf1.dist(p), self.sdf2.dist(p))
            }
        })+

        /// Get the intersection of two SDFs.
        #[derive(Clone, Copy, Debug)]
        pub struct Intersection<S1, S2, M> {
            pub sdf1: S1,
            pub sdf2: S2,
            pub max_func: M,
        }

        impl<S1, S2, M> Intersection<S1, S2, M> {
            /// Get the intersection of two SDFs.
            pub fn new(sdf1: S1, sdf2: S2, max_func: M) -> Self {
                Intersection {
                    sdf1,
                    sdf2,
                    max_func,
                }
            }
        }

        impl<S1, S2> Intersection<S1, S2, HardBlend> {
            pub fn hard(sdf1: S1, sdf2: S2) -> Self {
                Intersection {
                    sdf1,
                    sdf2,
                    max_func: HardBlend::default(),
                }
            }
        }

        $(impl<S1, S2> Intersection<S1, S2, PolySmoothBlend<$t>> {
            pub fn smooth(sdf1: S1, sdf2: S2, smoothness: $t) -> Self {
                Intersection {
                    sdf1,
                    sdf2,
                    max_func: PolySmoothBlend::<$t>::new(smoothness),
                }
            }
        }


        impl<V, S1, S2, M> Sdf<$t, V> for Intersection<S1, S2, M>
        where
            V: Vec<$t>,
            S1: Sdf<$t, V>,
            S2: Sdf<$t, V>,
            M: MaxFunction<$t>,
        {
            fn dist(&self, p: V) -> $t {
                self.max_func.max(self.sdf1.dist(p), self.sdf2.dist(p))
            }
        })+
    };
}

impl_ops!(f32, f64);

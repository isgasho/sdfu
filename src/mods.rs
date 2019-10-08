//! Modifiers for SDFs.
use super::*;
use crate::mathtypes::*;

macro_rules! impl_mods {
    ($($t:ty),+) => {
        /// Make an SDF have rounded outside edges.
        #[derive(Clone, Copy, Debug)]
        pub struct Round<T, S> {
            pub sdf: S,
            pub radius: T,
        }

        impl<T, S> Round<T, S> {
            pub fn new(sdf: S, radius: T) -> Self {
                Round { sdf, radius }
            }
        }

        $(impl<V, S> Sdf<$t, V> for Round<$t, S>
        where
            V: Vec<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                self.sdf.dist(p) - self.radius
            }
        })+

        /// Elongate an SDF along a single axis. The elongation is
        /// symmetrical around the origin.
        #[derive(Clone, Copy, Debug)]
        pub struct Elongate<T, S, D> {
            pub sdf: S,
            pub axis: Axis,
            pub elongation: T,
            _pd: std::marker::PhantomData<D>,
        }

        impl<T, S, D> Elongate<T, S, D> {
            /// Elongate an SDF along a single axis by `elongation`.
            pub fn new(sdf: S, axis: Axis, elongation: T) -> Self {
                Elongate {
                    sdf,
                    axis,
                    elongation,
                    _pd: std::marker::PhantomData,
                }
            }
        }

        $(impl<V, S> Sdf<$t, V> for Elongate<$t, S, Dim3D>
        where
            V: Vec3<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let h = match self.axis {
                    Axis::X => V::new(self.elongation, 0.0, 0.0),
                    Axis::Y => V::new(0.0, self.elongation, 0.0),
                    Axis::Z => V::new(0.0, 0.0, self.elongation),
                };
                let q = p - p.clamp(-h, h);
                self.sdf.dist(q)
            }
        }

        impl<V, S> Sdf<$t, V> for Elongate<$t, S, Dim2D>
        where
            V: Vec2<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let h = match self.axis {
                    Axis::X => V::new(self.elongation, 0.0),
                    Axis::Y => V::new(0.0, self.elongation),
                    Axis::Z => panic!("Attempting to use Z axis to elongate 2d SDF"),
                };
                let q = p - p.clamp(-h, h);
                self.sdf.dist(q)
            }
        })+

        /// Elongate an SDF along multiple axes.
        #[derive(Clone, Copy, Debug)]
        pub struct ElongateMulti<V, S, D> {
            pub sdf: S,
            pub elongation: V,
            _pd: std::marker::PhantomData<D>,
        }

        impl<V, S, D> ElongateMulti<V, S, D> {
            pub fn new(sdf: S, elongation: V) -> Self {
                ElongateMulti {
                    sdf,
                    elongation,
                    _pd: std::marker::PhantomData,
                }
            }
        }

        $(impl<V, S> Sdf<$t, V> for ElongateMulti<V, S, Dim3D>
        where
            V: Vec3<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let q = p.abs() - self.elongation;
                let t = q.y().max(q.z()).max(q.x()).min(0.0);
                self.sdf.dist(q.max(V::zero())) + t
            }
        }

        impl<V, S> Sdf<$t, V> for ElongateMulti<V, S, Dim2D>
        where
            V: Vec2<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let q = p.abs() - self.elongation;
                let t = q.x().max(q.y()).min(0.0);
                self.sdf.dist(q.max(V::zero())) + t
            }
        })+

        /// Translate an SDF.
        #[derive(Clone, Copy, Debug)]
        pub struct Translate<V, S> {
            pub sdf: S,
            pub translation: V,
        }

        impl<V, S> Translate<V, S> {
            pub fn new(sdf: S, translation: V) -> Self {
                Translate { sdf, translation }
            }
        }

        impl<T, V, S> Sdf<T, V> for Translate<V, S>
        where
            T: Copy,
            V: Vec<T>,
            S: Sdf<T, V>,
        {
            fn dist(&self, p: V) -> T {
                self.sdf.dist(p - self.translation)
            }
        }

        /// Rotate an SDF.
        #[derive(Clone, Copy, Debug)]
        pub struct Rotate<R, S> {
            pub sdf: S,
            pub rotation: R,
        }

        impl<R, S> Rotate<R, S> {
            pub fn new(sdf: S, rotation: R) -> Self {
                Rotate { sdf, rotation }
            }
        }

        impl<T, V, R, S> Sdf<T, V> for Rotate<R, S>
        where
            T: Copy,
            V: Vec<T>,
            S: Sdf<T, V>,
            R: Rotation<V> + Copy,
        {
            fn dist(&self, p: V) -> T {
                self.sdf.dist(self.rotation.rotate_vec(p))
            }
        }

        /// Rotate an SDF.
        #[derive(Clone, Copy, Debug)]
        pub struct Scale<T, S> {
            pub sdf: S,
            pub scaling: T,
        }

        impl<T, S> Scale<T, S> {
            pub fn new(sdf: S, scaling: T) -> Self {
                Scale { sdf, scaling }
            }
        }

        $(impl<V, S> Sdf<$t, V> for Scale<$t, S>
        where
            V: Vec<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                self.sdf.dist(p / self.scaling) * self.scaling
            }
        })+

        /// Fold (aka mirror) an SDF about a plane.
        #[derive(Clone, Copy, Debug)]
        pub struct Fold<S, V> {
            pub sdf: S,
            pub plane_norm: V,
        }

        impl<S, V> Fold<S, V> {
            pub fn new(sdf: S, plane_norm: V) -> Self {
                Fold { sdf, plane_norm }
            }
        }

        $(impl<V, S> Sdf<$t, V> for Fold<S, V>
        where
            V: Vec<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let t = 2.0 * p.dot(self.plane_norm).min(0.0);
                let p = p - self.plane_norm * t;
                self.sdf.dist(p)
            }
        })+

        #[derive(Clone, Copy, Debug)]
        pub enum Plane {
            XY,
            XZ,
            YZ,
        }

        /// Fold (aka mirror) an SDF about an axis aligned plane.
        #[derive(Clone, Copy, Debug)]
        pub struct AxisAlignedFold<S> {
            pub sdf: S,
            pub plane: Plane,
        }

        impl<S> AxisAlignedFold<S> {
            pub fn new(sdf: S, plane: Plane) -> Self {
                AxisAlignedFold { sdf, plane }
            }
        }

        $(impl<V, S> Sdf<$t, V> for AxisAlignedFold<S>
        where
            V: Vec3<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let p = match self.plane {
                    Plane::XY => V::new(p.x(), p.y(), p.z().abs()),
                    Plane::XZ => V::new(p.x(), p.y().abs(), p.z()),
                    Plane::YZ => V::new(p.x().abs(), p.y(), p.z()),
                };
                self.sdf.dist(p)
            }
        })+

        /// The Box Fold is often used when creating Kalaedescopic IFS fractal
        /// systems, and is one of the primary two pieces of the Mandelbox fractal.
        /// It is defined as (pseudocode)
        /// ```
        /// foreach axis a,
        ///     if point.a.abs() > 1
        ///         point.a = (2.0 - point.a).abs()
        /// ```
        #[derive(Clone, Copy, Debug)]
        pub struct BoxFold<S> {
            pub sdf: S,
        }

        impl<S> BoxFold<S> {
            pub fn new(sdf: S) -> Self {
                BoxFold { sdf }
            }
        }

        $(impl<V, S> Sdf<$t, V> for BoxFold<S>
        where
            V: Vec3<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let x = if p.x().abs() > 1.0 {
                    2.0 * p.x().signum() - p.x()
                } else {
                    p.x()
                };
                let y = if p.y().abs() > 1.0 {
                    2.0 * p.y().signum() - p.y()
                } else {
                    p.y()
                };
                let z = if p.z().abs() > 1.0 {
                    2.0 * p.z().signum() - p.z()
                } else {
                    p.z()
                };
                let p = V::new(x, y, z);
                self.sdf.dist(p)
            }
        })+

        /// The Ball Fold is often used when creating Kalaedescopic IFS fractal
        /// systems, and is one of the primary two pieces of the Mandelbox fractal.
        /// It is defined as (pseudocode)
        /// ```
        /// if point.magnitude() < radius
        ///     point /= radius^2
        /// else if point.magnitude() < 1
        ///     point.normalize()
        /// ```
        #[derive(Clone, Copy, Debug)]
        pub struct BallFold<T, S> {
            pub sdf: S,
            pub rad_squared: T,
        }

        $(impl<S> BallFold<$t, S> {
            pub fn new(sdf: S, radius: $t) -> Self {
                BallFold {
                    sdf,
                    rad_squared: radius * radius,
                }
            }
        }

        impl<V, S> Sdf<$t, V> for BallFold<$t, S>
        where
            V: Vec3<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let ms = p.magnitude_squared();
                let p = if ms < self.rad_squared {
                    p / self.rad_squared
                } else if ms < 1.0 {
                    p.normalized()
                } else {
                    p
                };
                self.sdf.dist(p)
            }
        })+

        /// Repeat an SDF infintely on one or more axes
        #[derive(Clone, Copy, Debug)]
        pub struct RepeatInfinite<S, V> {
            pub sdf: S,
            pub period: V,
        }

        impl<S, V> RepeatInfinite<S, V> {
            pub fn new(sdf: S, period: V) -> Self {
                RepeatInfinite { sdf, period }
            }
        }

        $(impl<V, S> Sdf<$t, V> for RepeatInfinite<S, V>
        where
            V: Vec<$t>,
            S: Sdf<$t, V>,
        {
            fn dist(&self, p: V) -> $t {
                let half_period = self.period * 0.5;
                let p = (p + half_period).euc_mod(self.period) - half_period;
                self.sdf.dist(p)
            }
        })+

        /// Repeat an SDF inside a bound on one or more axes
        #[derive(Clone, Copy, Debug)]
        pub struct RepeatBounded<S, V> {
            pub sdf: S,
            pub repetition_period: V,
            pub repetition_bounds: V,
        }

        impl<S, V> RepeatBounded<S, V> {
            pub fn new(sdf: S, repetition_period: V, repetition_bounds: V) -> Self {
                RepeatBounded {
                    sdf,
                    repetition_period,
                    repetition_bounds,
                }
            }
        }

        impl<T, V, S> Sdf<T, V> for RepeatBounded<S, V>
        where
            T: Copy,
            V: Vec<T>,
            S: Sdf<T, V>,
        {
            fn dist(&self, p: V) -> T {
                let p = p
                    - (self.repetition_period
                        * (p / self.repetition_period)
                            .round()
                            .clamp(-self.repetition_bounds, self.repetition_bounds));
                self.sdf.dist(p)
            }
        }
    };
}

impl_mods!(f32, f64);

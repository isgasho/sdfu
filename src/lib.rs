//! # `sdfu` - Signed Distance Field Utilities
//!
//! This is a small crate designed to help when working with signed distance fields
//! in the context of computer graphics, especially ray-marching based renderers. Most
//! of what is here is based on [Inigo Quilez' excellent articles](http://www.iquilezles.org/www/index.htm).
//!
//! If you're using one of the more popular math libraries in Rust, then just enable
//! the corresponding feature and hopefully all the necessary traits are already implemented
//! for you so that you can just start passing in your `Vec3`s or whatever your lib calls them
//! and you're off to the races! If not, then you can implement the necessary traits in the
//! `mathtypes` module and still use this library with your own math lib.
//!
//! This crate is built around the central traits `Sdf` and `SdfExt`. These traits are structured in a similar way to
//! how `std::iter::Iterator` works. Anything that implements `Sdf` is able to return a distance from
//! a point to its distance field. SDFs can be combined, modified, and otherwise used for various tasks
//! by using the combinator methods on the `SdfExt` trait, or by directly using the structs that actually
//! implement those combinators.
//!
//! Most `SDF`s will be build up from one or more primitives being modified and combined together--the
//! distance fields in the `primitive` module provide good starting points for this.
//!
//! # Demo
//!
//! ![demo image](https://raw.githubusercontent.com/termhn/sdfu/master/demo.png)
//!
//! The image above was rendered with my own path tracing renderer, [`rayn`](https://github.com/termhn/rayn),
//! by leveraging `sdfu`. The SDF that is rendered above was created with the following code:
//!
//! ```rust
//! use sdfu::{Sdf, SdfExt};
//!
//! let sdf = sdfu::Sphere::new(0.45)
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.25, 0.25, 1.5)))
//!     .union_smooth(
//!         sdfu::Sphere::new(0.3).translate(Vec3::new(0.3, 0.3, 0.0)),
//!         0.1)
//!     .union_smooth(
//!         sdfu::Sphere::new(0.3).translate(Vec3::new(-0.3, 0.3, 0.0)),
//!         0.1)
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(-0.3, 0.3, 0.0)))
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.125, 0.125, 1.5)).translate(Vec3::new(0.3, 0.3, 0.0)))
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(1.5, 0.1, 0.1)).translate(Vec3::new(0.0, 0.3, 0.0)))
//!     .subtract(
//!         sdfu::Box::new(Vec3::new(0.2, 2.0, 0.2)))
//!     .translate(Vec3::new(0.0, 0.0, -1.0));
//! ```
pub mod mathtypes;
use mathtypes::*;
pub use mathtypes::{Dim2D, Dim3D, Dimension};
pub mod primitives;
pub use primitives::*;

pub mod util;
use util::*;
pub mod ops;
use ops::*;
pub mod mods;
use mods::*;

/// The core trait of this crate; an implementor of this trait is able
/// to take in a vector and return the min distance from that vector to
/// a distance field.
pub trait Sdf<T, V: Vec<T>>: Copy {
    /// Get distance from `p` to this SDF.
    fn dist(&self, p: V) -> T;
}

pub trait SdfExt<T, V: Vec<T>>: Copy {
    /// Estimate the normals of this SDF using the default `NormalEstimator`.
    ///
    /// `eps` is the amount to change the point by for each sample.
    /// 0.001 is a good default value to try; you will ideally vary this based on distance.
    fn normals(
        self,
        eps: T,
    ) -> EstimateNormal<T, V, Self, CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>>
    where
        CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension>: NormalEstimator<T, V> + Default;

    /// Estimate the normals of this SDF using a fast, `TetrahedralEstimator`. Only
    /// works for 3d SDFs.
    ///
    /// `eps` is the amount to change the point by for each sample.
    /// 0.001 is a good default value to try; you will ideally vary this based on distance.
    fn normals_fast(self, eps: T) -> EstimateNormal<T, V, Self, TetrahedralEstimator<T, V>>
    where
        TetrahedralEstimator<T, V>: NormalEstimator<T, V> + Default;

    /// Estimate the normals of this SDF using a provided `NormalEstimator`.
    fn normals_with<E: NormalEstimator<T, V>>(self, estimator: E) -> EstimateNormal<T, V, Self, E>;

    /// Get the union of this SDF and another one using a standard
    /// hard minimum, creating a sharp crease at the boundary between the
    /// two fields.
    fn union<O: Sdf<T, V>>(self, other: O) -> Union<Self, O, HardBlend>;

    /// Get the union of this SDF and another one, blended together
    /// with a smooth minimum function. This uses a polynomial smooth min
    /// function by default, and the smoothing factor is controlled by the
    /// `smoothness` parameter. For even more control, see `union_with`.
    fn union_smooth<O: Sdf<T, V>>(
        self,
        other: O,
        softness: T,
    ) -> Union<Self, O, PolySmoothBlend<T>>;

    /// Get the union of this SDF and another one using a provided
    /// minimum function. See the documentation of `MinFunction` for more.
    fn union_with<O: Sdf<T, V>, M: MinFunction<T>>(
        self,
        other: O,
        min_function: M,
    ) -> Union<Self, O, M>;

    /// Get the subtracion of another SDF from this one. Note that this operation is *not* commutative,
    /// i.e. `a.subtraction(b) =/= b.subtraction(a)`.
    fn subtract<O: Sdf<T, V>>(self, other: O) -> Subtraction<Self, O, HardBlend>;

    /// Get the subtracion of another SDF from this one, smoothed by some factor `smoothness.`
    /// Note that this operation is *not* commutative,
    /// i.e. `a.subtraction(b) =/= b.subtraction(a)`.
    fn subtract_smooth<O: Sdf<T, V>>(
        self,
        other: O,
        smoothness: T,
    ) -> Subtraction<Self, O, PolySmoothBlend<T>>;

    /// Get the intersection of this SDF and another one.
    fn intersection<O: Sdf<T, V>>(self, other: O) -> Intersection<Self, O, HardBlend>;

    /// Get the intersection of this SDF and another one, smoothed by some factor `smoothness`.
    fn intersection_smooth<O: Sdf<T, V>>(
        self,
        other: O,
        smoothness: T,
    ) -> Intersection<Self, O, PolySmoothBlend<T>>;

    /// Round the corners of this SDF with a radius.
    fn round(self, radius: T) -> Round<T, Self>;

    /// Elongate this SDF along one axis. The elongation is symmetrical about the origin.
    fn elongate(self, axis: Axis, elongation: T) -> Elongate<T, Self, <V as Vec<T>>::Dimension>
    where
        Elongate<T, Self, <V as Vec<T>>::Dimension>: Sdf<T, V>;

    /// Elongate this SDF along one axis. The elongation is symmetrical about the origin.
    fn elongate_multi_axis(self, elongation: V) -> ElongateMulti<V, Self, <V as Vec<T>>::Dimension>
    where
        ElongateMulti<V, Self, <V as Vec<T>>::Dimension>: Sdf<T, V>;

    /// Translate the SDF by a vector.
    fn translate(self, translation: V) -> Translate<V, Self>;

    /// Rotate the SDF by a rotation.
    fn rotate<R: Rotation<V>>(self, rotation: R) -> Rotate<R, Self>;

    /// Scale the SDF by a uniform scaling factor.
    fn scale(self, scaling: T) -> Scale<T, Self>;
}

macro_rules! impl_sdf_ext {
    ($($t:ty),+) => {
        $(impl<S, V> SdfExt<$t, V> for S
            where S: Sdf<$t, V>,
                V: Vec<$t>,
        {
            fn normals(
                self,
                eps: $t,
            ) -> EstimateNormal<$t, V, Self, CentralDifferenceEstimator<$t, V, <V as Vec<$t>>::Dimension>>
            where
                CentralDifferenceEstimator<$t, V, <V as Vec<$t>>::Dimension>: NormalEstimator<$t, V> + Default,
            {
                EstimateNormal::new(self, CentralDifferenceEstimator::new(eps))
            }

            fn normals_fast(self, eps: $t) -> EstimateNormal<$t, V, Self, TetrahedralEstimator<$t, V>>
            where
                TetrahedralEstimator<$t, V>: NormalEstimator<$t, V> + Default,
            {
                EstimateNormal::new(self, TetrahedralEstimator::new(eps))
            }

            fn normals_with<E: NormalEstimator<$t, V>>(self, estimator: E) -> EstimateNormal<$t, V, Self, E> {
                EstimateNormal::new(self, estimator)
            }

            fn union<O: Sdf<$t, V>>(self, other: O) -> Union<Self, O, HardBlend> {
                Union::hard(self, other)
            }

            fn union_smooth<O: Sdf<$t, V>>(
                self,
                other: O,
                softness: $t,
            ) -> Union<Self, O, PolySmoothBlend<$t>> {
                Union::<_, _, PolySmoothBlend<$t>>::smooth(self, other, softness)
            }

            fn union_with<O: Sdf<$t, V>, M: MinFunction<$t>>(
                self,
                other: O,
                min_function: M,
            ) -> Union<Self, O, M> {
                Union::new(self, other, min_function)
            }

            fn subtract<O: Sdf<$t, V>>(self, other: O) -> Subtraction<Self, O, HardBlend> {
                Subtraction::hard(self, other)
            }

            fn subtract_smooth<O: Sdf<$t, V>>(
                self,
                other: O,
                smoothness: $t,
            ) -> Subtraction<Self, O, PolySmoothBlend<$t>> {
                Subtraction::<_, _, PolySmoothBlend<$t>>::smooth(self, other, smoothness)
            }

            fn intersection<O: Sdf<$t, V>>(self, other: O) -> Intersection<Self, O, HardBlend> {
                Intersection::hard(self, other)
            }

            fn intersection_smooth<O: Sdf<$t, V>>(
                self,
                other: O,
                smoothness: $t,
            ) -> Intersection<Self, O, PolySmoothBlend<$t>> {
                Intersection::<_, _, PolySmoothBlend<$t>>::smooth(self, other, smoothness)
            }

            fn round(self, radius: $t) -> Round<$t, Self> {
                Round::new(self, radius)
            }

            fn elongate(self, axis: Axis, elongation: $t) -> Elongate<$t, Self, <V as Vec<$t>>::Dimension>
            where
                Elongate<$t, Self, <V as Vec<$t>>::Dimension>: Sdf<$t, V>,
            {
                Elongate::new(self, axis, elongation)
            }

            fn elongate_multi_axis(self, elongation: V) -> ElongateMulti<V, Self, <V as Vec<$t>>::Dimension>
            where
                ElongateMulti<V, Self, <V as Vec<$t>>::Dimension>: Sdf<$t, V>,
            {
                ElongateMulti::new(self, elongation)
            }

            fn translate(self, translation: V) -> Translate<V, Self> {
                Translate::new(self, translation)
            }

            fn rotate<R: Rotation<V>>(self, rotation: R) -> Rotate<R, Self> {
                Rotate::new(self, rotation)
            }
            fn scale(self, scaling: $t) -> Scale<$t, Self> {
                Scale::new(self, scaling)
            }
        })+
    };
}

impl_sdf_ext!(f32, f64);

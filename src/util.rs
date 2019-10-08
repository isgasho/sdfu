//! Other random utilities that are helpful when using SDFs in computer graphics applications,
//! such as estimating normals.
use super::*;
use std::marker::PhantomData;

macro_rules! impl_util {
    ($($t:ty),+) => {
        /// `NormalEstimator`s provide a way to estimate the normal of the SDF `sdf` at point `p`.
        pub trait NormalEstimator<T, V: Vec<T>> {
            fn estimate_normal<S: Sdf<T, V>>(&self, sdf: S, p: V) -> V;
        }

        /// Estimates the normal of an `sdf` using an `estimator`, by default a `CentralDifferenceEstimator`,
        /// which provides a good default estimator that works for both 2D and 3D SDFs. See the documentation
        /// of `NormalEstimator` for more information.
        pub struct EstimateNormal<T, V, S, E> {
            pub sdf: S,
            pub estimator: E,
            _pd: PhantomData<(T, V)>,
        }

        impl<T, V, S, E> EstimateNormal<T, V, S, E> {
            /// Creates a new `EstimateNormal` with an SDF and a provided estimator.
            pub fn new(sdf: S, estimator: E) -> Self {
                EstimateNormal { sdf, estimator, _pd: PhantomData }
            }
        }

        $(impl<V, S, E> EstimateNormal<$t, V, S, E>
            where V: Vec<$t>,
                S: Sdf<$t, V>,
                E: NormalEstimator<$t, V>
        {
            /// Estimates the normal of the owned SDF at point p.
            pub fn normal_at(&self, p: V) -> V
            {
                self.estimator.estimate_normal(self.sdf, p)
            }
        })+


        /// Estimates the normal of an SDF by estimating the gradient of the SDF.
        ///
        /// The gradient is estimated by taking two samples of the SDF in each dimension,
        /// one slightly above (by `eps` distance) the point in question and one slightly below it and taking their
        /// difference, hence the 'central difference'. This estimation is relatively robust and accurate, and can
        /// work in both two and three dimensions, but is also relatively slow since it takes 6 samples of the SDF.
        /// See the `TetrahedralEstimator` for an estimator which is 3d only and slightly less robust/accurate but
        /// also slightly faster.
        ///
        /// See [this article](http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
        /// for more.
        pub struct CentralDifferenceEstimator<T, V, D> {
            pub eps: T,
            _pd: PhantomData<(V, D)>,
        }

        impl<T, V: Vec<T>> CentralDifferenceEstimator<T, V, <V as Vec<T>>::Dimension> {
            /// Creates a `CentralDifferenceEstimator` with a given epsilon value.
            pub fn new(eps: T) -> Self {
                CentralDifferenceEstimator {
                    eps,
                    _pd: std::marker::PhantomData,
                }
            }
        }

        $(impl<V> NormalEstimator<$t, V> for CentralDifferenceEstimator<$t, V, Dim3D>
        where
            V: Vec3<$t>,
        {
            fn estimate_normal<S: Sdf<$t, V>>(&self, sdf: S, p: V) -> V {
                let eps = self.eps;
                V::new(
                    sdf.dist(V::new(p.x() + eps, p.y(), p.z()))
                        - sdf.dist(V::new(p.x() - eps, p.y(), p.z())),
                    sdf.dist(V::new(p.x(), p.y() + eps, p.z()))
                        - sdf.dist(V::new(p.x(), p.y() - eps, p.z())),
                    sdf.dist(V::new(p.x(), p.y(), p.z() + eps))
                        - sdf.dist(V::new(p.x(), p.y(), p.z() - eps)),
                )
                .normalized()
            }
        }

        impl<V> NormalEstimator<$t, V> for CentralDifferenceEstimator<$t, V, Dim2D>
        where
            V: Vec2<$t>,
        {
            fn estimate_normal<S: Sdf<$t, V>>(&self, sdf: S, p: V) -> V {
                let eps = self.eps;
                V::new(
                    sdf.dist(V::new(p.x() + eps, p.y())) - sdf.dist(V::new(p.x() - eps, p.y())),
                    sdf.dist(V::new(p.x(), p.y() + eps)) - sdf.dist(V::new(p.x(), p.y() - eps)),
                )
                .normalized()
            }
        }

        impl<V: Vec<$t>> Default
            for CentralDifferenceEstimator<$t, V, <V as Vec<$t>>::Dimension>
        {
            fn default() -> Self {
                Self::new(0.0001)
            }
        })+

        /// Estimates the normal of an SDF by estimating the gradient of the SDF.
        ///
        /// The gradient is estimated by taking four samples of the SDF in a tetrahedron around the
        /// point of interest. By doing so, it only needs to take four instead of 6 samples of the SDF,
        /// like the CentralDifferenceEstimator does, so it is slightly faster. However, it only works
        /// for 3d SDFs and it is slightly less robust than the traditional way.
        ///
        /// See [this article](http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm)
        /// for more.
        pub struct TetrahedralEstimator<T, V> {
            pub eps: T,
            _pd: std::marker::PhantomData<V>,
        }

        impl<T, V> TetrahedralEstimator<T, V> {
            /// Creates a `TetrahedralEstimator` with a given epsilon value.
            pub fn new(eps: T) -> Self {
                TetrahedralEstimator {
                    eps,
                    _pd: std::marker::PhantomData,
                }
            }
        }

        $(impl<V> NormalEstimator<$t, V> for TetrahedralEstimator<$t, V>
        where
            V: Vec3<$t>,
        {
            fn estimate_normal<S: Sdf<$t, V>>(&self, sdf: S, p: V) -> V {
                let xyy = V::new(1.0, -1.0, -1.0);
                let yyx = V::new(-1.0, -1.0, 1.0);
                let yxy = V::new(-1.0, 1.0, -1.0);
                let xxx = V::new(1.0, 1.0, 1.0);

                (xyy * sdf.dist(p + xyy * self.eps)
                    + yyx * sdf.dist(p + xyy * self.eps)
                    + yxy * sdf.dist(p + xyy * self.eps)
                    + xxx * sdf.dist(p + xxx * self.eps))
                .normalized()
            }
        }

        impl<V: Vec<$t>> Default for TetrahedralEstimator<$t, V> {
            fn default() -> Self {
                Self::new(0.0001)
            }
        })+

    };
}

impl_util!(f32, f64);

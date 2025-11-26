"""
    IteratedExtendedKalmanFilter(kf, dynamics, measurement; Ajac, Cjac, step, maxiters, epsilon)
    IteratedExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=SimpleMvNormal(Matrix(R1)); nu::Int, ny=size(R2,1), Cjac = nothing, step = 1.0, maxiters=10, epsilon=1e-8)

A nonlinear state estimator propagating uncertainty using linearization. Returns an `ExtendedKalmanFilter` object but with Gauss-Newton based iterating measurement correction step.

The constructor to the iterated version of extended Kalman filter takes dynamics and measurement functions, and either covariance matrices, or a [`KalmanFilter`](@ref). If the former constructor is used, the number of inputs to the system dynamics, `nu`, must be explicitly provided with a keyword argument.

By default, the filter will internally linearize the dynamics using ForwardDiff. User provided Jacobian functions can be provided as keyword arguments `Ajac` and `Cjac`. These functions should have the signature `(x,u,p,t)::AbstractMatrix` where `x` is the state, `u` is the input, `p` is the parameters, and `t` is the time.

The dynamics and measurement function are of the following form
```
x(t+1) = dynamics(x, u, p, t) + w
y      = measurement(x, u, p, t) + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`

- `step` is the step size for the Gauss-Newton iterations. Float between 0 and 1. Default is 1.0 which should be good enough for most applications. For more challenging applications, a smaller step size might be necessary.
- `maxiters` is the maximum number of iterations. Default is 10. Usually a small number of iterations is needed. If higher number is needed, consider using UKF.
- `epsilon` is the convergence criterion. Default is 1e-8


See also [`UnscentedKalmanFilter`](@ref) which is more robust than `IteratedExtendedKalmanFilter`. See [`KalmanFilter`](@ref) for detailed instructions on how to set up a Kalman filter `kf`.
"""
IteratedExtendedKalmanFilter

function IteratedExtendedKalmanFilter(args...; kwargs...)
    return ExtendedKalmanFilter(args...; kwargs...)
end

function IteratedExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=SimpleMvNormal(Matrix(R1)); nu::Int, ny=size(R2,1), Cjac = nothing, step = 1.0, maxiters=10, epsilon=1e-8, kwargs...)
    IPM = !has_oop(measurement)
    T = promote_type(eltype(R1), eltype(R2), eltype(d0))
    nx = size(R1,1)
    measurement_model = IEKFMeasurementModel{T, IPM}(measurement, R2; nx, ny, Cjac, step, maxiters, epsilon)
    return ExtendedKalmanFilter(dynamics, measurement_model, R1, d0; nu, kwargs...)
end

function IteratedExtendedKalmanFilter(kf, dynamics, measurement; Ajac = nothing, Cjac = nothing, step = 1.0, maxiters = 10, epsilon = 1e-8, names=default_names(kf.nx, kf.nu, kf.ny, "IEKF"))
    IPD = !has_oop(dynamics)
    if measurement isa AbstractMeasurementModel
        measurement_model = measurement
        IPM = isinplace(measurement_model)
    else
        IPM = has_ip(measurement)
        T = promote_type(eltype(kf.R1), eltype(kf.R2), eltype(kf.d0))
        measurement_model = IEKFMeasurementModel{T, IPM}(measurement, kf.R2; kf.nx, kf.ny, Cjac, step, maxiters, epsilon)
    end
    if Ajac === nothing
        if IPD
            outx = zeros(eltype(kf.d0), kf.nx)
            jacx = zeros(eltype(kf.d0), kf.nx, kf.nx)
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian!(jacx, (xd,x)->dynamics(xd,x,u,p,t), outx, x)
        else
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
        end
    end

    return ExtendedKalmanFilter{IPD,IPM,typeof(kf),typeof(dynamics),typeof(measurement_model),typeof(Ajac)}(kf, dynamics, measurement_model, Ajac, names)
end


function correct!(kf::AbstractKalmanFilter,  measurement_model::IEKFMeasurementModel{IPM}, u, y, p = parameters(kf), t::Real = index(kf); R2 = get_mat(measurement_model.R2, kf.x, u, p, t), R12 = measurement_model.R12 === nothing ? nothing : get_mat(measurement_model.R12, kf.x, u, p, t)) where IPM
    (; x,R) = kf
    (; measurement, Cjac, step, maxiters, epsilon) = measurement_model

    xi = copy(x)

    if IPM
        pred_err = zeros(length(y))
        measurement(pred_err, xi, u, p, t)
        pred_err .= y .- pred_err
    else
        pred_err = y .- measurement(xi, u, p, t)
    end
    e = copy(pred_err)

    i = 1
    while true
        prev = xi
        C = Cjac(xi, u, p, t)
        if i > 1
            # We already evaluated e above
            if IPM
                measurement(e, xi, u, p, t)
                e .= y .- e
            else
                @bangbang e .= y .- measurement(xi, u, p, t)
            end
        end
        if R12 !== nothing
            # Simon's "Optimal State Estimation" Section 7.1 (Eq. 7.14)
            CR12 = C*R12
            S = symmetrize(C*R*C' + CR12 + CR12') + R2
            Sᵪ  = cholesky(Symmetric(S); check=false)
            issuccess(Sᵪ) || error("Cholesky factorization of innovation covariance failed, got S = $(printarray(S))")
            K = (R*C' + R12)/Sᵪ
        else
            S = symmetrize(C*R*C') + R2
            Sᵪ  = cholesky(Symmetric(S); check=false)
            issuccess(Sᵪ) || error("Cholesky factorization of innovation covariance failed, got S = $(printarray(S))")
            K = (R*C')/Sᵪ
        end
        dx = x-xi
        xi += vec(step*(dx + K*(e-C*dx)))
        # @show i, sum(abs, xi-prev)
        if sum(abs, xi-prev) < epsilon || i >= maxiters
            kf.x = xi
            if R12 !== nothing
                kf.R = symmetrize((I - K*C)*R - K*R12')
            else
                kf.R = symmetrize((I - K*C)*R) # WARNING against I .- A
            end
            ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), pred_err)[]# - 1/2*logdet(S) # logdet is included in logpdf
            e = pred_err
            return (; ll, e, S, Sᵪ, K)
        end
        i += 1
    end
end
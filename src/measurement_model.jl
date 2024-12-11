abstract type AbstractMeasurementModel end

struct ComponsiteMeasurementModel{M} <: AbstractMeasurementModel
    models::M
end

struct UKFMeasurementModel{IPM,AUGM,MT,RT,IT,MET,CT,CCT,CAT} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    ne::Int
    innovation::IT
    mean::MET
    cov::CT
    cross_cov::CCT
    cache::CAT
end

isinplace(::UKFMeasurementModel{IPM}) where IPM = IPM

UKFMeasurementModel{IPM,AUGM}(
    measurement,
    R2,
    ny,
    ne,
    innovation,
    mean,
    cov,
    cross_cov,
    cache = nothing,
) where {IPM,AUGM} = UKFMeasurementModel{
    IPM,
    AUGM,
    typeof(measurement),
    typeof(R2),
    typeof(innovation),
    typeof(mean),
    typeof(cov),
    typeof(cross_cov),
    typeof(cache),
}(
    measurement,
    R2,
    ny,
    ne,
    innovation,
    mean,
    cov,
    cross_cov,
    cache,
)


function add_cache(model::UKFMeasurementModel{IPM,AUGM}, cache) where {IPM,AUGM}
    UKFMeasurementModel{eltype(model.cache),IPM,AUGM}(
        model.measurement,
        model.R2,
        model.ny,
        model.ne,
        model.innovation,
        model.mean,
        model.cov,
        model.cross_cov,
        cache,
    )
end


function UKFMeasurementModel{T,IPM,AUGM}(
    measurement,
    R2;
    nx,
    ny,
    ne = nothing,
    innovation = -,
    mean = safe_mean,
    cov = safe_cov,
    cross_cov = cross_cov,
    static = nothing,
) where {T,IPM,AUGM}

    ne = if ne === nothing
        if !AUGM
            0
        elseif R2 isa AbstractArray
            size(R2, 1)
        else
            error(
                "The number of measurement noise variables, ne, can not be inferred from R2 when R2 is not an array, please provide the keyword argument `ne`.",
            )
        end
    else
        if AUGM && R2 isa AbstractArray && size(R2, 1) != ne
            error(
                "R2 must be square with size equal to the measurement vector length for non-augmented measurement",
            )
        end
    end
    if AUGM
        L = nx + ne
    else
        L = nx
    end
    static2 = something(static, L < 50 && !IPM)
    correct_sigma_point_cahce = SigmaPointCache{T}(nx, ne, ny, L, static2)
    UKFMeasurementModel{
        IPM,
        AUGM,
        typeof(measurement),
        typeof(R2),
        typeof(innovation),
        typeof(mean),
        typeof(cov),
        typeof(cross_cov),
        typeof(correct_sigma_point_cahce),
    }(
        measurement,
        R2,
        ny,
        ne,
        innovation,
        mean,
        cov,
        cross_cov,
        correct_sigma_point_cahce,
    )
end



struct SigmaPointCache{X0, X1}
    x0::X0
    x1::X1
end

function SigmaPointCache{T}(nx, nw, ny, L, static) where T
    if static
        x0 = [@SVector zeros(T, nx + nw) for _ = 1:2L+1]
        x1 = [@SVector zeros(T, ny) for _ = 1:2L+1]
    else
        x0 = [zeros(T, nx + nw) for _ = 1:2L+1]
        x1 = [zeros(T, ny) for _ = 1:2L+1]
    end
    SigmaPointCache(x0, x1)
end

Base.eltype(spc::SigmaPointCache) = eltype(spc.x0)


## EKF measurement model =======================================================

struct EKFMeasurementModel{IPM,MT,RT,CJ,CAT} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    Cjac::CJ
    cache::CAT
end

isinplace(::EKFMeasurementModel{IPM}) where IPM = IPM

EKFMeasurementModel{IPM}(
    measurement,
    R2,
    ny,
    Cjac,
    cache = nothing,
) where {IPM} = EKFMeasurementModel{
    IPM,
    typeof(measurement),
    typeof(R2),
    typeof(Cjac),
    typeof(cache),
}(
    measurement,
    R2,
    ny,
    Cjac,
    cache,
)


function add_cache(model::EKFMeasurementModel{IPM}, cache) where {IPM}
    EKFMeasurementModel{eltype(model.cache),IPM}(
        model.measurement,
        model.R2,
        model.ny,
        model.Cjac,
        cache,
    )
end


function EKFMeasurementModel{T,IPM}(
    measurement::M,
    R2;
    nx,
    ny,
    Cjac = nothing,
) where {T,IPM,M}

    
    if Cjac === nothing
        if IPM
            outy = zeros(T, ny)
            jacy = zeros(T, ny, nx)
            Cjac = (x,u,p,t) -> ForwardDiff.jacobian!(jacy, (y,x)->measurement(y,x,u,p,t), outy, x)
        else
            Cjac = (x,u,p,t) -> ForwardDiff.jacobian(x->measurement(x,u,p,t), x)
        end
    end


    EKFMeasurementModel{
        IPM,
        typeof(measurement),
        typeof(R2),
        typeof(Cjac),
        typeof(nothing),
    }(
        measurement,
        R2,
        ny,
        Cjac,
        nothing,
    )
end
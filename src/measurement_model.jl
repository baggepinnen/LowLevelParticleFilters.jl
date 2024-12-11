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
    cache,
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

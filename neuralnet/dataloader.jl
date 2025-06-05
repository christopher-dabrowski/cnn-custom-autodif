struct DataLoader
    X::Matrix{Float32}
    y::Matrix{Float32}
    batchsize::Int
    shuffle::Bool
    n_samples::Int
end

function DataLoader(X::AbstractMatrix, y::AbstractMatrix, batchsize::Int; shuffle::Bool=true)
    nx, ny = size(X, 2), size(y, 2)
    nx == ny || error("Inconsistent number of samples: X has $nx samples, y has $ny")
    n_samples = nx
    DataLoader(X, y, batchsize, shuffle, n_samples)
end
function Base.iterate(dl::DataLoader, state=(1, nothing))
    n = dl.n_samples
    start_idx, perm = state
    start_idx > n && return nothing

    if perm === nothing
        perm = dl.shuffle ? randperm(n) : 1:n
    end    
    
    end_idx = min(start_idx + dl.batchsize - 1, n)
    batch_indices = perm[start_idx:end_idx]
    
    x_batch = dl.X[:, batch_indices]
    
    y_batch = dl.y[:, batch_indices]
    
    next_state = (end_idx + 1, perm)
    return ((x_batch, y_batch), next_state)
end
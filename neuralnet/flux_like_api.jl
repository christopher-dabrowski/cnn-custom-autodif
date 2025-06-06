import Flux: glorot_uniform

abstract type Layer end

mutable struct Dense{F} <: Layer
  weight::Variable
  bias::Variable
  activation::F
  # TODO: Add name field for layer identification

  function Dense(in::Integer, out::Integer, activation=identity; bias=true, init=glorot_uniform)
    W = Variable(init(out, in), name="weight")
    b = bias ? Variable(zeros(out), name="bias") : nothing
    new{typeof(activation)}(W, b, activation)
  end
end

function (d::Dense)(x::GraphNode)
  if d.bias !== nothing
    return d.activation.(d.weight * x .+ d.bias)
  else
    return d.activation.(d.weight * x)
  end
end

function trainable(layer::Dense)::Vector{Variable}
  params = [layer.weight]
  if layer.bias !== nothing
    push!(params, layer.bias)
  end
  return params
end

struct Chain
  layers::Tuple
  function Chain(xs...)
    new(xs)
  end
end

function (c::Chain)(x)
  for layer in c.layers
    x = layer(x)
  end
  return x
end

function Base.getindex(c::Chain, i::Int)
  return c.layers[i]
end

function trainable(model::Chain)::Vector{Variable}
  params = Variable[]
  for layer in model.layers
    append!(params, trainable(layer))
  end
  return params
end

mutable struct Embedding <: Layer
  weight::Variable
  function Embedding(vocab_size::Integer, embedding_dim::Integer; init=glorot_uniform)
    W = Variable(init(embedding_dim, vocab_size), name="embedding_weight")
    new(W)
  end
end

function (e::Embedding)(x::AbstractVector{<:Inf64`})
  # x: vector of indices (1-based)
  # Returns: matrix of size (embedding_dim, length(x)), each column is embedding for one index
  return e.weight.output[:, x]
end

# TODO: Ask if we should train the embedding weights
function trainable(layer::Embedding)
  return [layer.weight]
end

abstract type Optimizer end

struct Adam
  η::Float64
  β1::Float64
  β2::Float64
  ε::Float64
  Adam(η=0.001, β1=0.9, β2=0.999, ε=1e-8) = new(η, β1, β2, ε)
end

abstract type OptimizerState end

mutable struct AdamState <: OptimizerState
  adam::Adam
  state::Dict{Variable,Tuple{Array,Array,Int}}
end

function setup(adam::Adam, model::Chain)
  params = trainable(model)
  state = Dict{Variable,Tuple{Array,Array,Int}}()
  for param in params
    m = zeros(size(param.output))
    v = zeros(size(param.output))
    state[param] = (m, v, 0)
  end
  return AdamState(adam, state)
end

function update!(adam_state::AdamState, model::Chain)
  η = adam_state.adam.η

  β1 = adam_state.adam.β1
  β2 = adam_state.adam.β2
  ε = adam_state.adam.ε

  adam_state = adam_state.state

  for param in trainable(model)
    g = param.gradient
    if size(g) != size(param.output)
      g = mean(g; dims=2)
      g = dropdims(g; dims=2)
    end

    m, v, t = adam_state[param]
    t += 1

    m .= β1 .* m .+ (1 - β1) .* g
    v .= β2 .* v .+ (1 - β2) .* (g .^ 2)

    m_hat = m ./ (1 - β1^t)
    v_hat = v ./ (1 - β2^t)

    param.output .-= η .* m_hat ./ (sqrt.(v_hat) .+ ε)

    adam_state[param] = (m, v, t)
  end
end

export Dense, Chain, Embedding

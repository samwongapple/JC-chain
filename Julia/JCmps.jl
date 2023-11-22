using Plotly, LinearAlgebra, Combinatorics
using ITensors

function JCtensor(χ::Int64, t::Float64, pdim::Int64 = 2)::Tuple{Index{Int64},Index{Int64},Index{Int64},Index{Int64},ITensor}
    p1 = Index(pdim,"p_index_1")
    p2 = Index(pdim,"p_index_2")
    v1 = Index(χ,"v_index_1")
    v2 = Index(χ,"v_index_2")
    A0::Array{ComplexF64} = zeros((χ,χ))
    A1::Array{ComplexF64} = zeros((χ,χ))
    B0::Array{ComplexF64} = zeros((χ,χ))
    B1::Array{ComplexF64} = zeros((χ,χ))

    for n in range(1,χ-1)
        A0[n,n] = cos(t*sqrt(n))
        B1[n,n] = cos(t*sqrt(n-1))
        A1[n+1,n] = sin(t*sqrt(n))
        B0[n,n+1] = sin(t*sqrt(n-1))
    end
    A0[χ,χ] = cos(t*sqrt(χ))
    B1[χ,χ] = cos(t*sqrt(χ-1))

    U::Array{ComplexF64} = zeros((pdim,pdim,χ,χ))
    U[1,1,:,:] = A0
    U[2,1,:,:] = 1im*A1
    U[1,2,:,:] = 1im*B0
    U[2,2,:,:] = B1
    
    JCtensor::ITensor = ITensor(U,p1,p2,v1,v2)
    return p1,p2,v1,v2,JCtensor
end

p1,p2,v1,v2,JCtensor1 = JCtensor(4,10.0,2)

U,S,V = svd(JCtensor1,p1,v1)

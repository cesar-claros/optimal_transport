### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ cbde530a-2ffd-48df-ab36-674545a5dc01
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["Plots",
			"GR", 
			"Images",
	 		"FileIO",
			"ImageMagick",
			"ImageIO",
			"ImageTransformations",
			"LinearAlgebra",
			"Optim",
	 		"Arpack",
			"Flux",
			"Statistics",
			"Colors",
			"PlutoUI",
			"Random",
			"Zygote",
			"Distances",
			"Clustering",
			"OptimalTransport",
			"Tulip"
			])
end

# ╔═╡ bc53c4fa-c4ec-11eb-2781-63cf775569b7
begin
	using Images
	using FileIO
	using ImageMagick
	using ImageIO
	using ImageTransformations
	using LinearAlgebra
	using Optim
	using Arpack
	using Flux
	using Flux.Optimise: update!
	using Statistics
	using Plots
	using Colors
	using PlutoUI
	using Random
	using Zygote: @adjoint
	using Distances
	using Clustering
	using OptimalTransport
	using Tulip
end

# ╔═╡ e90053d5-08eb-485d-97e1-a4aabd9c9fc3
TableOfContents()

# ╔═╡ 92190028-8dc1-4291-ada9-82a89bc3d6c8
md"""
# Max-sliced Bures Distance for Interpreting Discrepancies
## Benchmark for the color transfer task using max-sliced Bures and Wasserstein 
**June, 2021**
"""

# ╔═╡ 3a414199-1b11-4003-bb60-cf10eed619c9
begin
	img1_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"
	img2_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/758px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"	
	img3_url = "https://raw.githubusercontent.com/ChengzijunAixiaoli/PPMM/master/picture/ocean_day.jpg"
	img4_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
	img5_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/T%C3%BCbingen_-_Neckarfront_-_Ansicht_von_Eberhardsbr%C3%BCcke_in_Blauer_Stunde.jpg/1024px-T%C3%BCbingen_-_Neckarfront_-_Ansicht_von_Eberhardsbr%C3%BCcke_in_Blauer_Stunde.jpg"
	img6_url = "https://raw.githubusercontent.com/ChengzijunAixiaoli/PPMM/master/picture/ocean_sunset.jpg"
	imgs = [img1_url, img2_url, img3_url, img4_url, img5_url, img6_url]
	imgs_keys = ["Vassily Kandinsky", "Starry Night", "Ocean day", "Yellow labrador", "Blaue Stunde", "Ocean sunset"]
	selections = imgs_keys .=> imgs_keys
	lookup_element = Dict(imgs_keys .=> imgs)
	md"""
	Source image:$(@bind url_source_key Select(selections))
	
	Target image:$(@bind url_target_key Select(selections))
	"""
end

# ╔═╡ 87f0f40d-8db7-4628-8d66-b2cd753ebf7a
begin
	img_size = (512,512)
	url_source = lookup_element[url_source_key]
	url_target = lookup_element[url_target_key]
	source = imresize(load(download(url_source)), img_size)
	target = imresize(load(download(url_target)), img_size)
	mosaicview(source, target; nrow=1)
end

# ╔═╡ 77abc963-6ed3-47b9-8b57-9763c9bc866b
begin
	X = channelview(float64.(source))
	Y = channelview(float64.(target))
	color, height, width = size(X)
	X = reshape(X, (color,height*width))
	Y = reshape(Y, (color,height*width))
end

# ╔═╡ 9d917b0f-5b29-4222-8784-56a7347c68c1
@adjoint function sort(x)
	p = sortperm(x)
	x[p], x̄ -> (x̄[invperm(p)],)
end

# ╔═╡ 245bb8f2-1e85-4230-b2ae-32e821df9a9c
begin
	Eˢᵇ = (w,ρX,ρY) -> (√(w⋅(ρX*w))-√(w⋅(ρY*w)))/(w⋅w)
	Eˢʷ = (w,X,Y) -> norm(sort(X'*w)-sort(Y'*w), 2)
	cosine_dist = (Xₚ,Yₚ)->mean( (Xₚ*Yₚ')/(norm.(eachcol(Xₚ))⋅norm.(eachcol(Yₚ))) )
end

# ╔═╡ 656ceedb-4502-47e1-9f7c-53a4d46f5a14
function max_sliced(X, Y, ℓ, η=0.01)
	w = rand(size(X)[1])
	w = w/√(w⋅w)
	optimizer = ADAM(η)
	for iter in 1:100
		# Compute gradient of loss evaluated at w
        grads = gradient(params(w)) do
            return -ℓ(w,X,Y)
        end
		# @show -ℓ(w,X,Y)
        # Update parameters
		update!(optimizer, w, grads[w])
	end
	w = w/√(w⋅w)
end

# ╔═╡ b627ba4b-7b5b-4cd0-8cf2-b26430588b8e
function get_max_slice(X, Y, d_type="bures")
	if d_type == "bures"
		ℓ = Eˢᵇ
		ρX = X*X'
		ρY = Y*Y'
		wᵤᵥ = max_sliced(ρX, ρY, ℓ)
		wᵥᵤ = max_sliced(ρY, ρX, ℓ)
		if abs(ℓ(wᵤᵥ,ρX, ρY))<abs(ℓ(wᵥᵤ,ρX, ρY))
			w⃰ = wᵥᵤ
		else
			w⃰ = wᵤᵥ
		end
	elseif d_type == "wasserstein"
		ℓ = Eˢʷ
		w⃰ = max_sliced(X, Y, ℓ)
	end
	return w⃰
end

# ╔═╡ bd910797-65fd-4fe6-9098-29d05c32ad18
begin
	# imgs = [img1_url, img2_url, img3_url, img4_url, img5_url, img6_url]
	distance_keys = ["bures", "wasserstein"]
	d_selections = distance_keys .=> distance_keys
	
	transformation_keys = ["sliced ot", "affine", "shift"]
	t_selections = transformation_keys .=> transformation_keys
	# lookup_element = Dict(imgs_keys .=> imgs)
	md"""
	Distance :$(@bind distance_key Select(d_selections))
	
	Transformation :$(@bind transformation_key Select(t_selections))
	"""
end

# ╔═╡ 6da60eac-aa7f-4848-a3c8-29454bbce162
begin
	Random.seed!(12345)
	wᵦ = get_max_slice(X, Y, distance_key)
	Xₚ = X'*wᵦ
	Yₚ = Y'*wᵦ
	idx_vₛ = sortperm(Xₚ)
	idx_vₜ = sortperm(Yₚ)
	transfer = zeros(size(X))
	Δv = Yₚ[idx_vₜ] - Xₚ[idx_vₛ]
	transfer[:,idx_vₛ] = wᵦ*Δv'
	if transformation_key == "sliced ot"
		Xₜ = X + transfer	
	elseif transformation_key == "affine"
		μX = mean(X,dims=2)
		μY = mean(Y,dims=2)
		σX = std(X,dims=2)
		σY = std(Y,dims=2)
		Xₜ = (X.-μX).*(σY./σX).+μY + transfer
	elseif transformation_key == "shift"
		Δμ = mean(Y,dims=2) - mean(X,dims=2)
		Δσ = std(Y,dims=2)./std(X,dims=2)
		Xₜ = (X.+Δμ) + transfer
	end
	new_image = clamp01!(reshape(Xₜ, (color, height, width)))
	mosaicview(source, colorview(RGB, new_image), target; nrow=1)
end

# ╔═╡ fa428a7b-f1d7-4b7a-a6a4-c5cbb2adddc4
# begin
# 	Random.seed!(12345)
# 	nₖ = 500
# 	Cₛ = kmeans(X, nₖ; maxiter=250, display=:iter)
# 	Cₜ = kmeans(Y, nₖ; maxiter=250, display=:iter)
# 	Cₓ = kmeans(Xₜ, nₖ; maxiter=250, display=:iter)
# 	Cₛₜ = pairwise(SqEuclidean(), Cₛ.centers, Cₜ.centers; dims=2);
# 	Cₛₓ = pairwise(SqEuclidean(), Cₛ.centers, Cₓ.centers; dims=2);
# 	Cₓₜ = pairwise(SqEuclidean(), Cₓ.centers, Cₜ.centers; dims=2);
# 	@assert length(assignments(Cₓ))==length(assignments(Cₛ))==length(assignments(Cₜ))
# 	N = length(assignments(Cₓ))
# 	pmfₛ = counts(Cₛ)/N
# 	pmfₓ = counts(Cₓ)/N
# 	pmfₜ = counts(Cₜ)/N
# 	@show dₛₓ = emd2(pmfₛ, pmfₓ, Cₛₓ, Tulip.Optimizer())
# 	@show dₓₜ = emd2(pmfₓ, pmfₜ, Cₓₜ, Tulip.Optimizer())
# 	@show dₛₜ = emd2(pmfₛ, pmfₜ, Cₛₜ, Tulip.Optimizer())
# end

# ╔═╡ Cell order:
# ╠═cbde530a-2ffd-48df-ab36-674545a5dc01
# ╠═bc53c4fa-c4ec-11eb-2781-63cf775569b7
# ╠═e90053d5-08eb-485d-97e1-a4aabd9c9fc3
# ╠═92190028-8dc1-4291-ada9-82a89bc3d6c8
# ╠═3a414199-1b11-4003-bb60-cf10eed619c9
# ╠═87f0f40d-8db7-4628-8d66-b2cd753ebf7a
# ╠═77abc963-6ed3-47b9-8b57-9763c9bc866b
# ╠═9d917b0f-5b29-4222-8784-56a7347c68c1
# ╠═245bb8f2-1e85-4230-b2ae-32e821df9a9c
# ╠═656ceedb-4502-47e1-9f7c-53a4d46f5a14
# ╠═b627ba4b-7b5b-4cd0-8cf2-b26430588b8e
# ╠═bd910797-65fd-4fe6-9098-29d05c32ad18
# ╠═6da60eac-aa7f-4848-a3c8-29454bbce162
# ╠═fa428a7b-f1d7-4b7a-a6a4-c5cbb2adddc4

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

# ╔═╡ 51cebec0-6dc8-418e-b515-612284bf6191
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
			])
end

# ╔═╡ 3bb5ad9e-c64b-11eb-10bf-6fec84dec527
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
end

# ╔═╡ c9ccc5a2-7ac1-4101-bc1f-52cb3c3675c3
TableOfContents()

# ╔═╡ 14750130-5ba3-4630-ad7a-a09bfb7aa1e6
md"""
# Max-sliced Bures Distance for Interpreting Discrepancies
## Color transfer task using distributional sliced Bures and Wasserstein 
**July, 2021**
"""

# ╔═╡ 8e376434-6577-4a58-8017-66d9310550d6
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
	Source image:$(@bind url_source_key Select(selections; default=imgs_keys[6]))
	
	Target image:$(@bind url_target_key Select(selections; default=imgs_keys[5]))
	"""
end

# ╔═╡ 80f303a5-8de1-4158-af09-d164e34636eb
begin
	img_size = (512,512)
	url_source = lookup_element[url_source_key]
	url_target = lookup_element[url_target_key]
	source = imresize(load(download(url_source)), img_size)
	target = imresize(load(download(url_target)), img_size)
	mosaicview(source, target; nrow=1)
end

# ╔═╡ 5508fd26-287f-4a2c-883d-288659f1a0e2
begin
	X = channelview(float64.(source))
	Y = channelview(float64.(target))
	color, height, width = size(X)
	X = reshape(X, (color,height*width))
	Y = reshape(Y, (color,height*width))
end

# ╔═╡ 35f441e7-837a-423b-8b17-eaf522679d31
@adjoint function sort(x)
	p = sortperm(x)
	x[p], x̄ -> (x̄[invperm(p)],)
end

# ╔═╡ 4ed3711d-29b9-4561-9f2b-121266e8ea95
begin
	Eˢᵇ = (w,ρX,ρY) -> mean( (.√(eachcol(w).⋅eachcol(ρX*w))-.√(eachcol(w).⋅eachcol(ρY*w)))./(eachcol(w).⋅eachcol(w))) 
	Eˢʷ = (w,X,Y) -> mean( norm.( sort.(eachcol(X'*w))-sort.(eachcol(Y'*w)), 2) )
	cos_dist = (Xₚ,Yₚ)->mean( (Xₚ*Yₚ')/(norm.(eachcol(Xₚ))⋅norm.(eachcol(Yₚ))) )
end

# ╔═╡ 1d1f8007-84f4-4bf6-8f3e-d965a3817cdd
function distributional_sliced(X, Y, ℓ, nₚᵣₒⱼ, η=0.01, λ=1.0)
	fᵦ = Chain(Dense(size(X)[1], size(X)[1]), x->x/(norm(x,2)))
	# fᵦ = Dense(size(X)[1], size(X)[1])
	w = rand(size(X)[1], nₚᵣₒⱼ)
	optimizer = ADAM(η)
	for iter in 1:50
		proj = fᵦ(w)
		cᵩ = cos_dist(proj,proj)
		reg = λ*cᵩ
		# Compute gradient of loss evaluated at w
		grads = gradient(params(fᵦ)) do
			return -ℓ(fᵦ(w),X,Y)+reg
		end
		update!(optimizer, params(fᵦ), grads)
	end
	return proj = fᵦ(w)
end

# ╔═╡ ed1f3b23-e1c6-4677-8826-984b6fbcf05c
function get_distributional_slice(X, Y, nₚᵣₒⱼ, d_type="wasserstein")
	if d_type == "Bures"
		ℓ = Eˢᵇ
		ρX = X*X'
		ρY = Y*Y'
		wᵤᵥ = distributional_sliced(ρX, ρY, ℓ, nₚᵣₒⱼ)
		wᵥᵤ = distributional_sliced(ρY, ρX, ℓ, nₚᵣₒⱼ)
		if abs(ℓ(wᵤᵥ,ρX, ρY))<abs(ℓ(wᵥᵤ,ρX, ρY))
			w⃰ = wᵥᵤ
		else
			w⃰ = wᵤᵥ
		end
	elseif d_type == "Wasserstein"
		ℓ = Eˢʷ
		w⃰ = distributional_sliced(X, Y, ℓ, nₚᵣₒⱼ)
	end
	return w⃰
end

# ╔═╡ abe6ebdb-c446-4fdb-ac5c-669d3c8e36ba
begin
	# imgs = [img1_url, img2_url, img3_url, img4_url, img5_url, img6_url]
	distance_keys = ["Bures", "Wasserstein"]
	d_selections = distance_keys .=> distance_keys
	
	transformation_keys = ["sliced ot", "affine", "shift"]
	t_selections = transformation_keys .=> transformation_keys
	# lookup_element = Dict(imgs_keys .=> imgs)
	md"""
	Distance :$(@bind distance_key Select(d_selections))
	
	Transformation :$(@bind transformation_key Select(t_selections; default=transformation_keys[2]))
	"""
end

# ╔═╡ 934ecd21-2d1b-447f-917e-a2c27da81402
begin
	Random.seed!(12345)
	nₚᵣₒⱼ = 10
	wᵦ = get_distributional_slice(X, Y, nₚᵣₒⱼ, distance_key)
	X₀ = copy(X)
	Xₜ = zeros(size(X)[1],size(X)[2],nₚᵣₒⱼ)
	transfer = zeros(size(X))
	for i in 1:nₚᵣₒⱼ
		Xₚ = X₀'*wᵦ[:,i]
		Yₚ = Y'*wᵦ[:,i]
		idx_vₛ = sortperm(Xₚ)
		idx_vₜ = sortperm(Yₚ)
		Δv = Yₚ[idx_vₜ] - Xₚ[idx_vₛ]
 		transfer[:,idx_vₛ] = wᵦ[:,i]*Δv'
		if transformation_key == "sliced ot"
			Xₜ[:,:,i] = clamp01!(X₀ + transfer)	
		elseif transformation_key == "affine"
			μX = mean(X₀,dims=2)
			μY = mean(Y,dims=2)
			σX = std(X₀,dims=2)
			σY = std(Y,dims=2)
			Xₜ[:,:,i] = clamp01!((X₀.-μX).*(σY./σX).+μY + transfer)
		elseif transformation_key == "shift"
			Δμ = mean(Y,dims=2) - mean(X₀,dims=2)
			Δσ = std(Y,dims=2)./std(X₀,dims=2)
			Xₜ[:,:,i] = clamp01!((X₀.+Δμ) + transfer)
		end
		X₀ = Xₜ[:,:,i]
	end
	
	new_image = reshape(mean(Xₜ,dims=3), (color, height, width))
	mosaicview(source, colorview(RGB, new_image), target; nrow=1)
end

# ╔═╡ Cell order:
# ╟─51cebec0-6dc8-418e-b515-612284bf6191
# ╟─3bb5ad9e-c64b-11eb-10bf-6fec84dec527
# ╟─c9ccc5a2-7ac1-4101-bc1f-52cb3c3675c3
# ╟─14750130-5ba3-4630-ad7a-a09bfb7aa1e6
# ╟─8e376434-6577-4a58-8017-66d9310550d6
# ╠═80f303a5-8de1-4158-af09-d164e34636eb
# ╠═5508fd26-287f-4a2c-883d-288659f1a0e2
# ╠═35f441e7-837a-423b-8b17-eaf522679d31
# ╠═4ed3711d-29b9-4561-9f2b-121266e8ea95
# ╠═1d1f8007-84f4-4bf6-8f3e-d965a3817cdd
# ╠═ed1f3b23-e1c6-4677-8826-984b6fbcf05c
# ╟─abe6ebdb-c446-4fdb-ac5c-669d3c8e36ba
# ╠═934ecd21-2d1b-447f-917e-a2c27da81402

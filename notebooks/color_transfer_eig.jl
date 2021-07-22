### A Pluto.jl notebook ###
# v0.14.8

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

# ╔═╡ 02fa8807-b42b-4ac1-900c-bc504ddbac07
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add([
			Pkg.PackageSpec("Plots"),
			Pkg.PackageSpec("GR"), 
			Pkg.PackageSpec("Images"),
			Pkg.PackageSpec("FileIO"),
			Pkg.PackageSpec("ImageMagick"),
			Pkg.PackageSpec("ImageIO"),
			Pkg.PackageSpec("ImageTransformations"),
			Pkg.PackageSpec("LinearAlgebra"),
			Pkg.PackageSpec("Optim"),
			Pkg.PackageSpec("Arpack"),
			Pkg.PackageSpec(name="Flux", version="0.12.3"),
			Pkg.PackageSpec("Statistics"),
			Pkg.PackageSpec("Colors"),
			Pkg.PackageSpec("PlutoUI"),
			Pkg.PackageSpec("Random"),
			Pkg.PackageSpec(name="Zygote", version="0.6.14"),
			])
end

# ╔═╡ 7a0f179e-c416-11eb-3cdc-e129741c59b8
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

# ╔═╡ 4d90ea9e-1162-468d-9e63-17491b3a1941
TableOfContents()

# ╔═╡ 6deed45c-3fb3-4bc6-99bd-2aeb6a17a3cd
md"""
# Color transfer task using max-sliced Bures with eigenvector optimization 
**July, 2021**
"""

# ╔═╡ 80e7c955-69e0-40db-8c8e-97204be7842d
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

# ╔═╡ b5258974-4fa3-4e54-8366-46e7c9e17384
begin
	img_size = (512,512)
	url_source = lookup_element[url_source_key]
	url_target = lookup_element[url_target_key]
	source = imresize(load(download(url_source)), img_size)
	target = imresize(load(download(url_target)), img_size)
	mosaicview(source, target; nrow=1)
end

# ╔═╡ c1d78ec2-1f13-4dc8-8eef-240647a40eb5
begin
	X = channelview(float64.(source))
	Y = channelview(float64.(target))
	color, height, width = size(X)
	X = reshape(X, (color,height*width))
	Y = reshape(Y, (color,height*width))
	ρX = X*X'
	ρY = Y*Y'
end

# ╔═╡ bde3952a-8f61-46d6-b912-d90041994877
one_sided_bures_obj = (w,ρX,ρY) -> (√(w⋅(ρX*w))-√(w⋅(ρY*w)))/(w⋅w)

# ╔═╡ 2bc2ce9c-7949-4eb2-a128-224be25ca79c
function one_sided_MSB_eig(ρX, ρY)
	σI = 1e-7*Matrix{Float64}(I, size(ρX))
	eig_obj = w -> one_sided_bures_obj(w, ρX, ρY)
	get_eig = γ -> eigvecs(γ*ρX-ρY-σI)[:,1]
	# get_eig = γ -> eigs(γ*ρX-ρY-σI; nev=1)[2]
	f = γ -> -eig_obj(get_eig(γ))
	res = optimize(f,1e-6,1.0)
	γ⃰ = Optim.minimizer(res)
	w⃰ = get_eig(γ⃰)/(get_eig(γ⃰)⋅get_eig(γ⃰))
end

# ╔═╡ 231dd0be-8710-47a3-8760-caf768d0fbc6
function MSB_eig(ρX, ρY)
	wᵤᵥ = one_sided_MSB_eig(ρX, ρY)
	wᵥᵤ = one_sided_MSB_eig(ρY, ρX)
	if abs(one_sided_bures_obj(wᵤᵥ,ρX, ρY))<abs(one_sided_bures_obj(wᵥᵤ,ρX, ρY))
		w⃰ = wᵥᵤ
	else
		w⃰ = wᵤᵥ
	end
end

# ╔═╡ 1f59a5b2-4148-4d72-a024-e5628850107d
begin
	transformation_keys = ["sliced ot", "affine", "shift"]
	t_selections = transformation_keys .=> transformation_keys
	# lookup_element = Dict(imgs_keys .=> imgs)
	md"""
	Transformation :$(@bind transformation_key Select(t_selections; default=transformation_keys[2]))
	"""
end

# ╔═╡ 76be8f8c-8009-4198-bf6a-d74efdbe347d
begin
	wᵦ = MSB_eig(ρX, ρY)
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

# ╔═╡ Cell order:
# ╟─02fa8807-b42b-4ac1-900c-bc504ddbac07
# ╟─7a0f179e-c416-11eb-3cdc-e129741c59b8
# ╟─4d90ea9e-1162-468d-9e63-17491b3a1941
# ╟─6deed45c-3fb3-4bc6-99bd-2aeb6a17a3cd
# ╟─80e7c955-69e0-40db-8c8e-97204be7842d
# ╠═b5258974-4fa3-4e54-8366-46e7c9e17384
# ╠═c1d78ec2-1f13-4dc8-8eef-240647a40eb5
# ╠═bde3952a-8f61-46d6-b912-d90041994877
# ╠═2bc2ce9c-7949-4eb2-a128-224be25ca79c
# ╠═231dd0be-8710-47a3-8760-caf768d0fbc6
# ╟─1f59a5b2-4148-4d72-a024-e5628850107d
# ╠═76be8f8c-8009-4198-bf6a-d74efdbe347d

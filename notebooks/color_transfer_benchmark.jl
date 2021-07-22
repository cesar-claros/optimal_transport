### A Pluto.jl notebook ###
# v0.15.1

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

# ╔═╡ ae634b6a-e132-11eb-0bcd-df488fbf6110
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
			Pkg.PackageSpec("Clustering"),
			Pkg.PackageSpec("OptimalTransport"),
			Pkg.PackageSpec("Tulip"),
			Pkg.PackageSpec("LaTeXStrings"),
			Pkg.PackageSpec("BenchmarkTools"),
			])
end

# ╔═╡ ac9ffb10-8bd3-4079-ac67-c6c66285ff93
begin
	using Plots; gr()
	using Plots.PlotMeasures
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
	using Colors
	using PlutoUI
	using Random
	using Zygote: @adjoint
	using Clustering
	using OptimalTransport
	using Tulip
	using LaTeXStrings
	import BenchmarkTools
end

# ╔═╡ cf711bfa-5c1a-4f99-8167-930e1b06ff3a
TableOfContents()

# ╔═╡ 183acda2-3619-45ce-ac0e-1fe64da86ba7
md"""
# Color transfer algorithms benchmark 
**July, 2021**
"""

# ╔═╡ aa1ec4a8-c670-4091-b42e-bf303d4faec4
mutable struct data
	source::Matrix{RGB{N0f8}} # image type
	target::Matrix{RGB{N0f8}} # image type
	X::Matrix{Float64} # array representation of the source
	Y::Matrix{Float64} # array representation of the target
	color::Int64
	height::Int64
	width::Int64
	function data(source::Matrix{RGB{N0f8}}, target::Matrix{RGB{N0f8}})
		X = channelview(float64.(source))
		Y = channelview(float64.(target))
		color, height, width = size(X)
		X = reshape(X, (color,height*width))
		Y = reshape(Y, (color,height*width)) 
		new(source, target, X, Y, color, height, width)
	end
end

# ╔═╡ e12debaa-6b8e-4d72-ab7f-7cf3b6efc416
md"""
### Loading images and preprocess
Two different images (source and target) have to be chosen in order to perform the color transfer task. For this notebook, there are 6 possible choices for the images, which allows 15 different combinations that can be explored. These images are downloaded from the web, so their dimensions (height and width) are not the same. Since our procedure requires the same number of pixels for both the source and the target, we need to resize the images to the same dimensions. In this case, we chose $512\times512$. 
"""

# ╔═╡ 61826a81-bd68-4c35-8423-49e4d8acaaa9
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

# ╔═╡ 44a373c4-fb2d-449a-b32a-1d88ba03fc34
begin
	default(titlefont = (20, "times"), legend=false, showaxis=false, ticks=false)
	img_size = (512,512)
	url_source = lookup_element[url_source_key]
	url_target = lookup_element[url_target_key]
	source = imresize(load(download(url_source)), img_size)
	target = imresize(load(download(url_target)), img_size)
	plot(	plot(source, title="Source"), plot(target, title="Target"),
			layout = (1, 2), margin=-1mm, size=(700,400))
end

# ╔═╡ cf583ff5-dfc5-4933-b799-793117dc9790
function D_μν(X::Matrix{Float64}, Y::Matrix{Float64}; seed::Int64=12345, nₖ::Int64=500, iter::Int64=250)
	Random.seed!(seed)
	Cₛ = kmeans(X, nₖ; maxiter=iter)
	Cₜ = kmeans(Y, nₖ; maxiter=iter)
	Cₛₜ = pairwise(SqEuclidean(), Cₛ.centers, Cₜ.centers; dims=2);
	N = length(assignments(Cₛ))
	pmfₛ = counts(Cₛ)/N
	pmfₜ = counts(Cₜ)/N
	dₛₜ = emd2(pmfₛ, pmfₜ, Cₛₜ, Tulip.Optimizer())
	return (Cₛ,Cₜ,pmfₛ,pmfₜ,dₛₜ)
end

# ╔═╡ 70ab9c3b-459e-4e2f-918d-a2a6c656372d
begin
	d = data(source, target)
	Cₛ,Cₜ,pmfₛ,pmfₜ,dₛₜ = D_μν(d.X, d.Y)
end

# ╔═╡ faa85adf-2aa9-415f-9bce-223170ef40a6
begin
	dist = round(dₛₜ; digits=3)
	# text = L"d_{st}=%$dist"
	plot_font = "Computer Modern";
	l = @layout [ grid(2,2);
				a{0.01h}]
	plot(	plot(source, title="Source"),
			plot(target, title="Target"),
			scatter(Cₛ.centers[1,:],Cₛ.centers[3,:],color=colorview(RGB,Cₛ.centers),
					showaxis=true, ticks=true, xlim=(0,1), ylim=(0,1),
					ylabel="blue", xlabel="red"),
			scatter(Cₜ.centers[1,:],Cₜ.centers[3,:],color=colorview(RGB,Cₜ.centers),
					showaxis=true, yticks=false, xticks=true, xlim=(0,1), ylim=(0,1), 
					ylabel="", xlabel="red"),
			plot(;title=L"d_{st}=%$dist",titlefont=font(15,plot_font)),
			layout=l, margin=1mm, size=(700,800) )
end

# ╔═╡ b051df02-7661-4eba-a39e-082a883fba2f
@adjoint function sort(x)
	p = sortperm(x)
	x[p], x̄ -> (x̄[invperm(p)],)
end

# ╔═╡ 83534466-d37d-4382-962f-eea26aae25a2
begin
	♢ = eachcol
	Eˢᵇ(w::Matrix{Float64},ρX::Matrix{Float64},ρY::Matrix{Float64}) = mean( (.√(♢(w).⋅♢(ρX*w))-.√(♢(w).⋅♢(ρY*w)))./(♢(w).⋅♢(w)) ) 
	Eˢʷ(w::Matrix{Float64},X::Matrix{Float64},Y::Matrix{Float64}) = mean( norm.( sort.(♢(X'*w))-sort.(♢(Y'*w)), 2) )
	Eˢᵇ(w::Vector{Float64},ρX::Matrix{Float64},ρY::Matrix{Float64}) = mean( (.√(♢(w).⋅♢(ρX*w))-.√(♢(w).⋅♢(ρY*w)))./(♢(w).⋅♢(w)) ) 
	Eˢʷ(w::Vector{Float64},X::Matrix{Float64},Y::Matrix{Float64}) = mean( norm.( sort.(♢(X'*w))-sort.(♢(Y'*w)), 2) )
	cos_dist = (Xₚ::Matrix{Float64},Yₚ::Matrix{Float64}) -> mean( (Xₚ*Yₚ')/(norm.(♢(Xₚ))⋅norm.(♢(Yₚ))) )
end

# ╔═╡ 4b438cec-0197-45f3-93a7-3b85a1edc6f1
function optimizeSlice(X::Matrix{Float64}, Y::Matrix{Float64}, ℓ::Function, nₚᵣₒⱼ::Int64; η::Float64=0.01, λ::Float64=1.0, nᵢₜ::Int64=100)
	fᵦ = Chain(Dense(size(X)[1], size(X)[1]), x->x/(norm(x,2)))
	w = rand(size(X)[1], nₚᵣₒⱼ)
	optimizer = ADAM(η)
	for iter in 1:nᵢₜ
		# Compute gradient of loss evaluated at fᵦ(w)
		grads = gradient(Flux.params(fᵦ)) do
			return -ℓ(fᵦ(w),X,Y)+λ*cos_dist(fᵦ(w),fᵦ(w))
		end
		update!(optimizer, Flux.params(fᵦ), grads)
	end
	return proj = fᵦ(w)
end

# ╔═╡ 71b89994-02fd-49d6-a8a6-50f2e658e97a
function optimizeSlice(X::Matrix{Float64}, Y::Matrix{Float64}, ℓ::Function; η::Float64=0.01, nᵢₜ::Int64=100)
	w = rand(size(X)[1])
	w = w/√(w⋅w)
	optimizer = ADAM(η)
	for iter in 1:nᵢₜ
		# Compute gradient of loss evaluated at w
        grads = gradient(Flux.params(w)) do
            return -ℓ(w,X,Y)
        end
		update!(optimizer, w, grads[w])
	end
	w = w/√(w⋅w)
end

# ╔═╡ 98d6c4cf-1e8b-40df-8ea5-c27cbc819443
function optimizeSlice(ρX::Matrix{Float64}, ρY::Matrix{Float64})
	σI = 1e-7*Matrix{Float64}(I, size(ρX))
	eig_obj = w -> Eˢᵇ(w, ρX, ρY)
	get_eig = γ -> eigvecs(γ*ρX-ρY-σI)[:,1]
	# get_eig = γ -> eigs(γ*ρX-ρY-σI; nev=1)[2]
	f = γ -> -eig_obj(get_eig(γ))
	res = optimize(f,1e-6,1.0)
	γ⃰ = Optim.minimizer(res)
	return w⃰ = get_eig(γ⃰)/(get_eig(γ⃰)⋅get_eig(γ⃰))
end

# ╔═╡ c38ef6f2-fcaf-4174-9b58-f40addeea4ac
function getSlice(X::Matrix{Float64}, Y::Matrix{Float64}, nₚᵣₒⱼ::Int64, distance_type::String)
	if distance_type == "Bures"
		ℓ = Eˢᵇ
		ρX = X*X'
		ρY = Y*Y'
		wᵤᵥ = optimizeSlice(ρX,ρY,ℓ,nₚᵣₒⱼ)
		wᵥᵤ = optimizeSlice(ρY,ρX,ℓ,nₚᵣₒⱼ)
		if abs(ℓ(wᵤᵥ,ρX, ρY)) < abs(ℓ(wᵥᵤ,ρX, ρY))
			w⃰ = wᵥᵤ
		else
			w⃰ = wᵤᵥ
		end
	elseif distance_type == "Wasserstein"
		ℓ = Eˢʷ
		w⃰ = optimizeSlice(X, Y, ℓ, nₚᵣₒⱼ)
	end
	return w⃰
end

# ╔═╡ a22ae7b3-3114-4aa3-939e-53f52b9dfd93
function getSlice(X::Matrix{Float64}, Y::Matrix{Float64}, distance_type::String)
	if distance_type == "Bures"
		ℓ = Eˢᵇ
		ρX = X*X'
		ρY = Y*Y'
		wᵤᵥ = optimizeSlice(ρX,ρY,ℓ)
		wᵥᵤ = optimizeSlice(ρY,ρX,ℓ)
		if abs(ℓ(wᵤᵥ,ρX, ρY)) < abs(ℓ(wᵥᵤ,ρX, ρY))
			w⃰ = wᵥᵤ
		else
			w⃰ = wᵤᵥ
		end
	elseif distance_type == "Wasserstein"
		ℓ = Eˢʷ
		w⃰ = optimizeSlice(X, Y, ℓ)
	end
	return w⃰
end

# ╔═╡ fea0f977-feb6-4765-b897-660692331dbd
function getSlice(X::Matrix{Float64}, Y::Matrix{Float64})
	ρX = X*X'
	ρY = Y*Y'
	wᵤᵥ = optimizeSlice(ρX, ρY)
	wᵥᵤ = optimizeSlice(ρY, ρX)
	if abs(Eˢᵇ(wᵤᵥ,ρX, ρY))<abs(Eˢᵇ(wᵥᵤ,ρX, ρY))
		w⃰ = wᵥᵤ
	else
		w⃰ = wᵤᵥ
	end
	return w⃰
end

# ╔═╡ c8033c6e-931a-4475-ad01-72c6529a68b9
function getImage(X::Matrix{Float64}, Y::Matrix{Float64}, nₚᵣₒⱼ::Int64, getSlice::Function, distance_key::String, transformation_key::String; seed::Int64=12345)
	Random.seed!(seed)
	# nₚᵣₒⱼ = 1
	wᵦ = getSlice(X, Y, nₚᵣₒⱼ, distance_key)
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
	
	return mean(Xₜ,dims=3)
	# return new_image = reshape(mean(Xₜ,dims=3), (color, height, width))
	# mosaicview(source, colorview(RGB, new_image), target; nrow=1)
end

# ╔═╡ 39a17ec9-7bf8-4111-ac33-f572c677bb3a
function getImage(X::Matrix{Float64}, Y::Matrix{Float64}, getSlice::Function, distance_key::String, transformation_key::String; seed::Int64=12345)
	Random.seed!(seed)
	nₚᵣₒⱼ = 1
	wᵦ = getSlice(X, Y, distance_key)
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
	
	return mean(Xₜ,dims=3)
	# return new_image = reshape(mean(Xₜ,dims=3), (color, height, width))
	# mosaicview(source, colorview(RGB, new_image), target; nrow=1)
end

# ╔═╡ 9245f690-0a1b-4faa-8878-347354b9484f
function getImage(X::Matrix{Float64}, Y::Matrix{Float64}, getSlice::Function, transformation_key::String; seed::Int64=12345)
	Random.seed!(seed)
	nₚᵣₒⱼ = 1
	wᵦ = getSlice(X, Y)
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
	
	return mean(Xₜ,dims=3)
	# return new_image = reshape(mean(Xₜ,dims=3), (color, height, width))
	# mosaicview(source, colorview(RGB, new_image), target; nrow=1)
end

# ╔═╡ a2e9955a-ec62-4425-9550-a0117ee713fe
function D_μXν(Xₜ::Matrix{Float64}, Cₛ::KmeansResult, Cₜ::KmeansResult, pmfₛ::Vector{Float64}, pmfₜ::Vector{Float64}; nₖ::Int64=500, maxiter::Int64=250)
	Cₓ = kmeans(Xₜ, nₖ; maxiter)
	Cₛₓ = pairwise(SqEuclidean(), Cₛ.centers, Cₓ.centers; dims=2);
	Cₓₜ = pairwise(SqEuclidean(), Cₓ.centers, Cₜ.centers; dims=2);
	N = length(assignments(Cₛ))
	pmfₓ = counts(Cₓ)/N
	dₛₓ = emd2(pmfₛ, pmfₓ, Cₛₓ, Tulip.Optimizer())
	dₓₜ = emd2(pmfₓ, pmfₜ, Cₓₜ, Tulip.Optimizer())
	return dₛₓ, dₓₜ
end

# ╔═╡ 0b618cd6-d4e5-4e51-9ca6-631d61d01b44
begin
	types = ["max-eig", "distributional", "distributional", "max", "max"]
	d_keys = ["Bures", "Bures", "Wasserstein", "Bures", "Wasserstein"]
	n_proj = ["", 1, 1, "", ""]
	images = Array{Float64,3}[]
	distances = Tuple{Float64,Float64}[]
	times = String[]
	for i in 1:5
		if types[i]=="distributional"
			new_image = getImage(d.X, d.Y, n_proj[i], getSlice, d_keys[i], "affine")
			new_time = BenchmarkTools.@benchmark getSlice(d.X, d.Y, n_proj[$i], d_keys[$i])
		elseif types[i]=="max"
			new_image = getImage(d.X, d.Y, getSlice, d_keys[i], "affine")
			new_time = BenchmarkTools.@benchmark getSlice(d.X, d.Y, d_keys[$i])
		elseif types[i]=="max-eig"
			new_image = getImage(d.X, d.Y, getSlice, "affine")
			new_time = BenchmarkTools.@benchmark getSlice(d.X, d.Y)
		end
		new_time = BenchmarkTools.prettytime(mean(new_time).time)
		push!(times, new_time)
		new_distance = D_μXν(reshape(new_image, (d.color, d.height*d.width)), Cₛ, Cₜ, pmfₛ, pmfₜ)
		push!(distances, new_distance)
		new_image = reshape(new_image, (d.color, d.height, d.width))
		push!(images, new_image)
	end
end

# ╔═╡ a98ba841-05b7-482b-adc6-223b4c116494
begin
	lt = (@layout [° _ _ _ ° ; ° ° ° ° ° ; ° ° ° ° °{0.001h} ; ° ° ° ° °{0.001h} ; ° ° ° ° °{0.001h} ])
	# image_plots = [plot(colorview(RGB, images[i]), title=types[i]*"\n"*d_keys[i]) for i in 1:5]
	image_plots = map(i->plot(colorview(RGB,images[i]), title=types[i]*"\n"*d_keys[i]), 1:5)
	dsX_info = map(i->plot(title=L"d_{sX}=%$i"), map(j->round(distances[j][1];digits=3), 1:5))
	dXt_info = map(i->plot(title=L"d_{Xt}=%$i"), map(j->round(distances[j][2];digits=3), 1:5))
	t_info = map(i->plot(title=latexstring("t=",i)), times)
	plot(	plot(source, title="source"),
			plot(target, title="target"),
			image_plots...,
			dsX_info...,
			dXt_info...,
			t_info...,
			layout=lt, margin=0mm, size=(700,400), titlefont=font(10,plot_font) )
end

# ╔═╡ Cell order:
# ╟─ae634b6a-e132-11eb-0bcd-df488fbf6110
# ╠═ac9ffb10-8bd3-4079-ac67-c6c66285ff93
# ╟─cf711bfa-5c1a-4f99-8167-930e1b06ff3a
# ╟─183acda2-3619-45ce-ac0e-1fe64da86ba7
# ╠═aa1ec4a8-c670-4091-b42e-bf303d4faec4
# ╟─e12debaa-6b8e-4d72-ab7f-7cf3b6efc416
# ╟─61826a81-bd68-4c35-8423-49e4d8acaaa9
# ╠═44a373c4-fb2d-449a-b32a-1d88ba03fc34
# ╠═cf583ff5-dfc5-4933-b799-793117dc9790
# ╠═70ab9c3b-459e-4e2f-918d-a2a6c656372d
# ╠═faa85adf-2aa9-415f-9bce-223170ef40a6
# ╠═83534466-d37d-4382-962f-eea26aae25a2
# ╠═b051df02-7661-4eba-a39e-082a883fba2f
# ╠═4b438cec-0197-45f3-93a7-3b85a1edc6f1
# ╠═71b89994-02fd-49d6-a8a6-50f2e658e97a
# ╠═98d6c4cf-1e8b-40df-8ea5-c27cbc819443
# ╠═c38ef6f2-fcaf-4174-9b58-f40addeea4ac
# ╠═a22ae7b3-3114-4aa3-939e-53f52b9dfd93
# ╠═fea0f977-feb6-4765-b897-660692331dbd
# ╠═c8033c6e-931a-4475-ad01-72c6529a68b9
# ╠═39a17ec9-7bf8-4111-ac33-f572c677bb3a
# ╠═9245f690-0a1b-4faa-8878-347354b9484f
# ╠═a2e9955a-ec62-4425-9550-a0117ee713fe
# ╠═0b618cd6-d4e5-4e51-9ca6-631d61d01b44
# ╠═a98ba841-05b7-482b-adc6-223b4c116494

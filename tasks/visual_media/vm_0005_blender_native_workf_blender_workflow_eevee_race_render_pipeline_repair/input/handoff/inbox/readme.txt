RACE / ACCEPTANCE REGISTER
Frame 1; 480 x 270, 100%, square pixels. EEVEE 64 samples; source motion blur parameters, blur enabled. All/Ship/Set enabled with Combined and Depth and unchanged collection filtering. Transparent Film; compositing on, sequencer off. sRGB / Standard / None / exposure 0 / gamma 1.

Retain the shot's geometry, modifiers, material palettes, lighting, camera, world, animation and timeline. Restore original material definitions, UV layers and polygon material assignments from native remnants without replacing shot geometry or renaming its material identities. Store recovered surface data on retained base meshes. Restore and pack available image datablocks to catalog.csv identities and colorspaces. fx3_Panels_Bump2.jpg remains connected and missing.

repaired.blend contains a top-level linked compositor group instance Delivery from finish.blend. Its unlinked scalar controls Gain/Near/Far/Epsilon are saved at 1.4/20/60/0.08. All library and export paths stay blend-relative inside output. Native internals are editable; equivalent graphs accepted. A normal still-output render regenerates all exports beside repaired.blend.

Finishing model, using untouched scene-linear passes:
A = All Combined RGBA; S = Ship Combined RGBA; Za/Zs = All/Ship Depth.
V = 1 if abs(Zs-Za) < Epsilon, else 0.
D = clamp((Far-Za)/(Far-Near),0,1); empty depth -> 0.
C = S.rgb / max(S.alpha,0.000001).
T_c(x) = a_c * max(x,0)^b_c, with positive, constant per-channel coefficients constrained by scan.csv (input/accepted_response pairs; pixel coordinates use top-left origin).
F.rgb = A.rgb + V*S.alpha*D*(Gain*T(C)-C); F.alpha = A.alpha.
The response applies to straight linear RGB; F is unclipped linear RGBA. Controls operate over Gain [0.5,1.8], Near [16,28], Far [45,70], Epsilon [0.01,0.2], Near < Far.

Exports, all 480 x 270:
beauty_0001.png: RGBA16, display A.
finish_0001.png: RGBA16, display F; actual Composite receives F.
ship_matte_0001.png: grayscale16, numeric S.alpha.
depth_0001.png: grayscale16, numeric D.
utility_0001.png: RGB16, numeric (V*S.alpha,S.alpha,D).
passes_0001.exr: lossless multilayer FLOAT32, original Combined RGBA and raw Depth for All/Ship/Set plus linear F RGBA. Names: <layer>.Combined.R/G/B/A; <layer>.Depth.V or .Z. F uses Finish.Combined.R/G/B/A or Composite.Combined.R/G/B/A. Extra channels allowed.
Only display PNGs use the scene view transform. Data PNGs and EXR remain numeric/linear. Both files are reopened elsewhere and rerendered with controls, exposure and light energies changed.

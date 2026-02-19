
def rect(x, y, w, h):
    return f"M {x} {y} h {w} v {h} h {-w} z"

def rounded_rect(x, y, w, h, r):
    # Simplified
    return f"M {x+r} {y} h {w-2*r} a {r} {r} 0 0 1 {r} {r} v {h-2*r} a {r} {r} 0 0 1 {-r} {r} h {-w+2*r} a {r} {r} 0 0 1 {-r} {-r} v {-h+2*r} a {r} {r} 0 0 1 {r} {-r} z"

def polygon(points):
    return "M " + " ".join([f"{p[0]} {p[1]}" for p in points]) + " z"

# Center X for Front is 150
CX_F = 150
# Center X for Back is 450
CX_B = 450

svg_parts = []
ids = []

# --- FRONT ---
# Head (generic base, not clickable itself, parts are Jaw)
# Actually for WPI: "Jaw" (Left/Right)
# We will make the "Cheeks/Jaw" clickable.
# Head base
svg_parts.append(f'<circle cx="{CX_F}" cy="50" r="30" fill="#eee" />')
# Jaw Left/Right
svg_parts.append(f'<path id="jaw-left" class="body-part" d="{polygon([(CX_F-5, 70), (CX_F-25, 65), (CX_F-25, 50), (CX_F-5, 50)])}" />')
svg_parts.append(f'<path id="jaw-right" class="body-part" d="{polygon([(CX_F+5, 70), (CX_F+25, 65), (CX_F+25, 50), (CX_F+5, 50)])}" />')

# Neck
svg_parts.append(f'<path id="neck" class="body-part" d="{rect(CX_F-15, 80, 30, 20)}" />')

# Chest (Upper Torso)
svg_parts.append(f'<path id="chest" class="body-part" d="{rect(CX_F-40, 100, 80, 50)}" />')

# Abdomen (Lower Torso)
svg_parts.append(f'<path id="abdomen" class="body-part" d="{rect(CX_F-35, 150, 70, 60)}" />')

# Shoulders (L/R)
svg_parts.append(f'<path id="shoulder-left" class="body-part" d="{polygon([(CX_F-40, 100), (CX_F-70, 105), (CX_F-65, 130), (CX_F-40, 125)])}" />')
svg_parts.append(f'<path id="shoulder-right" class="body-part" d="{polygon([(CX_F+40, 100), (CX_F+70, 105), (CX_F+65, 130), (CX_F+40, 125)])}" />')

# Upper Arms
svg_parts.append(f'<path id="upper-arm-left" class="body-part" d="{polygon([(CX_F-70, 105), (CX_F-90, 110), (CX_F-80, 160), (CX_F-65, 130)])}" />')
svg_parts.append(f'<path id="upper-arm-right" class="body-part" d="{polygon([(CX_F+70, 105), (CX_F+90, 110), (CX_F+80, 160), (CX_F+65, 130)])}" />')

# Lower Arms
svg_parts.append(f'<path id="lower-arm-left" class="body-part" d="{polygon([(CX_F-80, 160), (CX_F-90, 210), (CX_F-75, 215), (CX_F-65, 165)])}" />')
svg_parts.append(f'<path id="lower-arm-right" class="body-part" d="{polygon([(CX_F+80, 160), (CX_F+90, 210), (CX_F+75, 215), (CX_F+65, 165)])}" />')

# Upper Legs
svg_parts.append(f'<path id="upper-leg-left" class="body-part" d="{rect(CX_F-35, 210, 30, 80)}" />')
svg_parts.append(f'<path id="upper-leg-right" class="body-part" d="{rect(CX_F+5, 210, 30, 80)}" />')

# Lower Legs
svg_parts.append(f'<path id="lower-leg-left" class="body-part" d="{rect(CX_F-30, 290, 20, 80)}" />')
svg_parts.append(f'<path id="lower-leg-right" class="body-part" d="{rect(CX_F+10, 290, 20, 80)}" />')


# --- BACK ---
# Head Base
svg_parts.append(f'<circle cx="{CX_B}" cy="50" r="30" fill="#eee" />')

# Upper Back (Trapezzius/Center)
svg_parts.append(f'<path id="upper-back" class="body-part" d="{rect(CX_B-40, 100, 80, 60)}" />')

# Lower Back
svg_parts.append(f'<path id="lower-back" class="body-part" d="{rect(CX_B-35, 160, 70, 50)}" />')

# Hips/Glutes (L/R) - Important for Fibro
svg_parts.append(f'<path id="hip-left" class="body-part" d="{rect(CX_B-35, 210, 35, 40)}" />')
svg_parts.append(f'<path id="hip-right" class="body-part" d="{rect(CX_B, 210, 35, 40)}" />')

# Same limbs for back pretty much, mapped to logic if needed, but WPI doesn't distinguish L/R back arms usually, but let's just duplicate visual structure but maybe map to same IDs?
# WPI just says "Upper arm right", doesn't say front/back.
# So we should use SAME IDs for front/back limbs so selecting one selects the other?
# YES. That's better UX.
# Redefining IDs to match Front for limbs.

# Shoulders (Back view) - mapped to same ID as front?
# Actually users might feel pain in back of shoulder vs front. WPI is specific: "Shoulder girdle".
# It usually implies the joint area. Let's map both front/back visuals to the SAME logic ID.
# To do that, I'll give them different DOM IDs but the same "data-region" attribute.

def part(pid, region, d):
    return f'<path id="{pid}" data-region="{region}" class="body-part" d="{d}" />'

svg_final = []
# REBUILDING WITH DATA-REGION

# FRONT
svg_final.append(f'<text x="{CX_F}" y="20" text-anchor="middle" font-weight="bold">FRONT</text>')
svg_final.append(f'<circle cx="{CX_F}" cy="50" r="30" fill="#e0e0e0" />') # Head decoration
svg_final.append(part("f-jaw-l", "jaw-left", polygon([(CX_F-5, 70), (CX_F-25, 65), (CX_F-25, 50), (CX_F-5, 50)])))
svg_final.append(part("f-jaw-r", "jaw-right", polygon([(CX_F+5, 70), (CX_F+25, 65), (CX_F+25, 50), (CX_F+5, 50)])))
svg_final.append(part("f-neck", "neck", rect(CX_F-15, 80, 30, 20)))
svg_final.append(part("f-chest", "chest", rect(CX_F-40, 100, 80, 50)))
svg_final.append(part("f-abdomen", "abdomen", rect(CX_F-35, 150, 70, 60)))
svg_final.append(part("f-sh-l", "shoulder-left", polygon([(CX_F-40, 100), (CX_F-70, 105), (CX_F-65, 130), (CX_F-40, 125)])))
svg_final.append(part("f-sh-r", "shoulder-right", polygon([(CX_F+40, 100), (CX_F+70, 105), (CX_F+65, 130), (CX_F+40, 125)])))
svg_final.append(part("f-ua-l", "upper-arm-left", polygon([(CX_F-70, 105), (CX_F-90, 110), (CX_F-80, 160), (CX_F-65, 130)])))
svg_final.append(part("f-ua-r", "upper-arm-right", polygon([(CX_F+70, 105), (CX_F+90, 110), (CX_F+80, 160), (CX_F+65, 130)])))
svg_final.append(part("f-la-l", "lower-arm-left", polygon([(CX_F-80, 160), (CX_F-90, 210), (CX_F-75, 215), (CX_F-65, 165)])))
svg_final.append(part("f-la-r", "lower-arm-right", polygon([(CX_F+80, 160), (CX_F+90, 210), (CX_F+75, 215), (CX_F+65, 165)])))
svg_final.append(part("f-ul-l", "upper-leg-left", rect(CX_F-35, 210, 30, 80)))
svg_final.append(part("f-ul-r", "upper-leg-right", rect(CX_F+5, 210, 30, 80)))
svg_final.append(part("f-ll-l", "lower-leg-left", rect(CX_F-30, 290, 20, 80)))
svg_final.append(part("f-ll-r", "lower-leg-right", rect(CX_F+10, 290, 20, 80)))

# BACK
svg_final.append(f'<text x="{CX_B}" y="20" text-anchor="middle" font-weight="bold">BACK</text>')
svg_final.append(f'<circle cx="{CX_B}" cy="50" r="30" fill="#e0e0e0" />')
svg_final.append(part("b-neck", "neck", rect(CX_B-15, 80, 30, 20))) # Back of neck -> "Neck"
svg_final.append(part("b-ub", "upper-back", rect(CX_B-40, 100, 80, 60)))
svg_final.append(part("b-lb", "lower-back", rect(CX_B-35, 160, 70, 50)))
svg_final.append(part("b-hip-l", "hip-left", rect(CX_B-35, 210, 35, 40))) # Glutes
svg_final.append(part("b-hip-r", "hip-right", rect(CX_B, 210, 35, 40)))

# Back Shoulders - mapping to same Shoulder regions
svg_final.append(part("b-sh-l", "shoulder-left", polygon([(CX_B-40, 100), (CX_B-70, 105), (CX_B-65, 130), (CX_B-40, 125)])))
svg_final.append(part("b-sh-r", "shoulder-right", polygon([(CX_B+40, 100), (CX_B+70, 105), (CX_B+65, 130), (CX_B+40, 125)])))
# Back Arms - mapping to same Upper Arm regions
svg_final.append(part("b-ua-l", "upper-arm-left", polygon([(CX_B-70, 105), (CX_B-90, 110), (CX_B-80, 160), (CX_B-65, 130)])))
svg_final.append(part("b-ua-r", "upper-arm-right", polygon([(CX_B+70, 105), (CX_B+90, 110), (CX_B+80, 160), (CX_B+65, 130)])))
svg_final.append(part("b-la-l", "lower-arm-left", polygon([(CX_B-80, 160), (CX_B-90, 210), (CX_B-75, 215), (CX_B-65, 165)])))
svg_final.append(part("b-la-r", "lower-arm-right", polygon([(CX_B+80, 160), (CX_B+90, 210), (CX_B+75, 215), (CX_B+65, 165)])))
# Back Legs - Upper Legs (Hamstrings) -> same Upper Leg
svg_final.append(part("b-ul-l", "upper-leg-left", rect(CX_B-35, 250, 30, 40))) # Just lower part of glutes
svg_final.append(part("b-ul-r", "upper-leg-right", rect(CX_B+5, 250, 30, 40)))
# Back Lower Legs -> same Lower Leg
svg_final.append(part("b-ll-l", "lower-leg-left", rect(CX_B-30, 290, 20, 80)))
svg_final.append(part("b-ll-r", "lower-leg-right", rect(CX_B+10, 290, 20, 80)))



with open('body_map.svg', 'w') as f:
    f.write(f'<svg viewBox="0 0 600 400" id="body-map-svg">\n')
    f.write('\n'.join(svg_final))
    f.write('\n</svg>')
print("SVG saved to body_map.svg")

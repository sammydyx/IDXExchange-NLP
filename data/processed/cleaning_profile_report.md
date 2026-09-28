# Week 2 Cleaning Profiling Report

## What the raw remarks require
- Listings profiled: 1,000
- Null rate: 0.00%
- Average raw remark length: 1337 characters
- Listings with HTML-like tags: 1
- Listings with compact price mentions (`k` or `m`): 13

## Cleaning decisions
- Decode HTML and remove tags.
- Normalize Unicode punctuation and whitespace.
- Expand MLS abbreviations such as `br`, `ba`, `sqft`, `w/`, and `a/c`.
- Standardize compact prices and square-footage measurements.

## Detected abbreviations
br, bdrm, bd, ba, sqft, sq ft, sf, w/, w/o, a/c, ac, hvac, gar, kit, lr, dr, mbr, mb, hoa, ss, app, upd, renov, yr, yrs, approx, incl, det

## Most common raw tokens
- `and`: 9472
- `the`: 8605
- `with`: 3657
- `to`: 3417
- `of`: 2947
- `in`: 2926
- `home`: 2539
- `for`: 2447
- `this`: 2264
- `living`: 1920
- `is`: 1783
- `an`: 1750
- `room`: 1287
- `or`: 1168
- `space`: 1123
- `offers`: 1070
- `kitchen`: 944
- `private`: 927
- `dining`: 818
- `features`: 797

## Before and after examples

Each excerpt is centered on the first changed portion of a real listing remark.

### Listing `1157753806`

**Before:** e front home is big enough for a small or medium family. It offers 3 bedrooms and 1 bathroom (approx. 1,120 sq ft) and was partially updated within the past year. It was rented for $2300. The rear home includes 1 bedroom and 1 bathroom (approx. 595 sq ft), fully updated about 3 months ago. The 2-car garage attached to...

**After:** e front home is big enough for a small or medium family. It offers 3 bedrooms and 1 bathroom (approximately. 1120 square feet) and was partially updated within the past year. It was rented for $2300. The rear home includes 1 bedroom and 1 bathroom (approximately 595 square feet), fully updated about 3 months ago. The 2-car garage attached to...

### Listing `1177906910`

**Before:** Tucked within one of De Luz’s most coveted enclaves, where custom estates are quietly hidden among ancient oaks just minutes from Old Town Temecula, this remarkable estate was designed to create a place where life’s most meaningful moments simply

**After:** Tucked within one of De Luz's most coveted enclaves, where custom estates are quietly hidden among ancient oaks just minutes from Old Town Temecula, this remarkable estate was designed to create a place where life's most meaningful moments simply

### Listing `1189804557`

**Before:** room is equipped with surround sound and flows seamlessly into the main living areas. Upstairs, you’ll find generously sized bedrooms along with a large loft/theater room perfect for movie nights, gaming, or an additional living space. The backyard is a true private oasis and built for Southern California living, feat

**After:** room is equipped with surround sound and flows seamlessly into the main living areas. Upstairs, you'll find generously sized bedrooms along with a large loft/theater room perfect for movie nights, gaming, or an additional living space. The backyard is a true private oasis and built for Southern California living, feat

### Listing `1156723303`

**Before:** maintenance, each unit offers modern finishes including 9' & 10' ceilings, Milgard windows, central A/C, stainless steel appliances, tankless water heaters, in-unit laundry and private outdoor storage. The property has 8 on-site parking spaces with 4 steel carports, individually metered for gas and electric and sub-met

**After:** maintenance, each unit offers modern finishes including 9' & 10' ceilings, Milgard windows, central air conditioning, stainless steel appliances, tankless water heaters, in-unit laundry and private outdoor storage. The property has 8 on-site parking spaces with 4 steel carports, individually metered for gas and electri

### Listing `1168721531`

**Before:** eatown with panoramic Downtown LA views! Built in 2020, this modern 3-bedroom, 2-bath condo offers 1,403 sq ft of bright, open living space with only one shared wall and low HOA dues. The south/east-facing layout is filled with natural light and features an open-concept living and dining area with recessed LED lighting

**After:** eatown with panoramic Downtown LA views! Built in 2020, this modern 3-bedroom, 2-bath condo offers 1403 square feet of bright, open living space with only one shared wall and low homeowners association dues. The south/east-facing layout is filled with natural light and features an open-concept living and dining area wi


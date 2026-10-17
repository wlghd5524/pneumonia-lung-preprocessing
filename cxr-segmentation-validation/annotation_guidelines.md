# Annotation guidelines

These notes describe the procedure used for the 100 AP expert-reference contours. Structure-by-structure inclusion rules for individual anatomical landmarks were not written down before annotation. The contour followed the specialist's judgment of the visible lung field.

## Annotators

One annotator contoured every image. The annotator is a study author whose specialty is infectious disease. These contours are that specialist's reference. They are not contours drawn or reviewed by a radiologist. A second reader did not contour the same images, did not revise them, and inter-reader agreement was not measured. Years in practice are not recorded in the annotation files, so this guideline does not state a number of years.

## Annotation software

The contours were drawn in a custom in-house browser program written for this study. The program has no separate release version. It served the radiograph in a local browser, required a login before any radiograph was shown, and displayed a review number rather than the DICOM identifier. Polygons were rasterized with Pillow.

The contouring screen showed the original chest radiograph and a review number. The report-derived pneumonia label, the prespecified challenging stratum, and portable-acquisition status were withheld on that screen. The same program has a separate page, linked from the contouring header, that can show the original image together with the MedSAM3 and CheXMask lung images of these 100 cases. The manuscript describes that page as outside the annotation session. This repository has no access log that shows whether the page was opened while the contours were drawn, so the procedure is not described here as a verified blind.

The same program could open 200 upright PA radiographs with an existing lung overlay as a shape reference. Those 200 images are a different set from the 100 AP radiographs that were contoured.

The annotator placed points along a lung edge. Clicking the first point closed that region. A brush stroke that formed a closed loop was filled. Polygon fill and brush fill were combined by a pixel-wise union. Brightness and contrast changed the display only. In the saved reference set, every mask consists of two closed polygon regions, and no brush layer was stored.

## Anatomical boundaries

The annotator drew the left and right lung fields as separate closed regions on the original radiograph. Each region was stored as a polygon in native image coordinates, filled, and combined by union into one mask.

The recorded instructions do not define a separate rule for the lung apex, the hilum, the diaphragm, or the cardiac border. Placement of those edges was the specialist's judgment of the visible lung field on that radiograph. Points were stored in image pixels, with the origin at the top-left corner, x increasing to the right, and y increasing downward. A point outside the image was moved onto the nearest image edge before the polygon was filled.

## Pathological regions

The recorded instructions do not state whether consolidation or pleural fluid is included inside the lung mask. Where the edge was difficult to see, the annotator still placed a contour and could mark the case as having a difficult lung boundary.

On the same reading, the annotator answered four yes-or-no questions from the radiograph alone:

| Question | Meaning used on the annotation screen |
| --- | --- |
| Rotation | The body was photographed at a tilt. |
| Low inspiration | The ribs are poorly expanded and the lungs look small. |
| Difficult lung boundary | The boundary is indistinct enough to make delineation difficult. |
| Pneumonia | The radiograph appears to show pneumonia. The dataset label was not shown. |

Free-text notes were optional. Notes in this sample mention tilt, a redaction box, a monitoring line over part of one lung, and one lower border that was difficult to place because of pleural fluid. A note records that difficulty. It does not encode a general rule for including or excluding pleural fluid.

## Mask format

Each case was saved as one 8-bit grayscale PNG. Lung pixels are 255 and background pixels are 0. The mask has the same width and height as the source radiograph. In evaluation, every pixel with a gray value greater than 0 is lung. The two filled lung regions form that single mask. There is no left-right label in the PNG.

A polygon file was stored beside each PNG during annotation. Its schema is a JSON object with a `regions` list. Each region is a list of `[x, y]` points in image pixels. Those polygon files name individual radiographs and are not part of this public release. The PNG is the mask used for Dice, IoU, and HD95.

## Quality control

Saving a contour did not require the four questions. Marking a case verified did. Verification required at least one closed lung region and an answer of yes or no to all four questions. A verified contour was locked: points, brush, questions, and saving stayed disabled until verification was cleared. Clearing verification kept the saved mask and allowed editing.

The program asked for confirmation when the filled lung covered less than 1 percent of the image pixels. All 100 saved masks contain two regions, all 100 were marked verified, and none has a stored brush layer.

A separate sheet for an explicit judgment that a MedSAM3 or CheXMask mask was wrong was left blank. The segmentation scores use the expert contour directly. No second reader reviewed or revised the contours.

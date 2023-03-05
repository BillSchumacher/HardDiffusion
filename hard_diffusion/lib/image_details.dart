import 'dart:convert';
import 'dart:ui';

import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:hard_diffusion/api/images/generated_image.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:provider/provider.dart';
import 'package:super_clipboard/super_clipboard.dart';

final Image errorImage = Image.asset("assets/error.gif");
final Image paintingImage = Image.asset("assets/painting.gif");

class ImageDetails extends StatefulWidget {
  const ImageDetails({super.key, required this.item});

  final GeneratedImage item;
  @override
  _ImageDetailsState createState() => _ImageDetailsState(item: item);
}

class _ImageDetailsState extends State<ImageDetails> {
  _ImageDetailsState({required this.item});
  final GeneratedImage item;
  bool showEditButton = false;
  Uint8List imageBytes = Uint8List(0);

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var previewImage = appState.taskPreview[item.taskId];
    var currentStep = appState.taskCurrentStep[item.taskId];
    var image;
    var generating = false;
    var progress = 0.0;
    if (item.error!) {
      image = errorImage;
    } else if (item.generatedAt == null) {
      if (previewImage != null && previewImage.isNotEmpty) {
        final decodedBytes = base64Decode(previewImage);
        imageBytes = decodedBytes;
        image = Image.memory(decodedBytes);
      } else {
        image = paintingImage;
      }
      generating = true;
      currentStep ??= 0;
      progress = currentStep / item.numInferenceSteps!;
    } else {
      image = CachedNetworkImage(
        placeholder: (context, url) => const CircularProgressIndicator(),
        imageUrl: "http://localhost:8000/${item.filename}",
        imageBuilder: (context, imageProvider) {
          imageProvider
              .obtainKey(createLocalImageConfiguration(context))
              .then((value) {
            imageProvider.load(value, (bytes,
                {allowUpscaling = true,
                cacheHeight = 200,
                cacheWidth = 200}) async {
              imageBytes = bytes.buffer.asUint8List();
              return instantiateImageCodec(imageBytes);
            });
            return Container(
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: imageProvider,
                  fit: BoxFit.fill,
                  //colorFilter: ColorFilter.mode(Colors.red, BlendMode.colorBurn)),
                ),
              ),
            );
          });

          return Image(
            image: imageProvider,
          );
        },
        errorWidget: (context, url, error) => errorImage,
      );
    }
    var contextSize = MediaQuery.of(context).size;
    var contextHeight = contextSize.height;
    var contextWidth = contextSize.width;
    var aspectRatio = contextWidth / contextHeight;
    return Stack(
      children: <Widget>[
        InkWell(
            onTap: () => setState(() => showEditButton = !showEditButton),
            child: image),
        if (generating && progress > 0) ...[
          Positioned(
            bottom: 0,
            child: Container(
              width: contextWidth,
              height: contextHeight * aspectRatio * 0.069,
              child: LinearProgressIndicator(
                value: progress * 0.22,
                semanticsLabel: 'Progress',
                minHeight: 1.0,
              ),
            ),
          ),
        ],
        if (showEditButton) ...[
          Positioned(
            bottom: 5,
            left: 5,
            child: ElevatedButton(
              child: Text('Copy'),
              onPressed: () async {
                final item = DataWriterItem();
                item.add(Formats.png(imageBytes));
                await ClipboardWriter.instance.write([item]);
              },
            ),
          ),
          Positioned(
            bottom: 5,
            right: 5,
            child: ElevatedButton(
              child: Text('Delete'),
              onPressed: () => {},
            ),
          ),
          Positioned(
            top: 5,
            left: 5,
            child: ElevatedButton(
              child: Text('Use as Input'),
              onPressed: () => {},
            ),
          ),
          Positioned(
            top: 5,
            right: 5,
            child: ElevatedButton(
              child: Text('Use settings'),
              onPressed: () => {},
            ),
          ),
        ]
      ],
    );
  }
}

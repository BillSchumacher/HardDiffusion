import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/infer_text_to_image.dart';
import 'package:hard_diffusion/generated_images.dart';
import 'package:hard_diffusion/vertical_separator.dart';

class GeneratePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.max,
      children: [
        Flexible(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text("Generate",
                    style: Theme.of(context).textTheme.titleLarge),
              ),
              InferTextToImageForm(),
            ],
          ),
        ),
        VerticalSeparator(),
        GeneratedImages(),
      ],
    );
  }
}

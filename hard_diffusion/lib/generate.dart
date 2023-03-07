import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/infer_text_to_image.dart';
import 'package:hard_diffusion/generated_images.dart';
import 'package:hard_diffusion/vertical_separator.dart';

class GeneratePage extends StatelessWidget {
  const GeneratePage({super.key, required this.landscape});
  final bool landscape;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.max,
      children: [
        Flexible(
            child: InferTextToImageForm(
          landscape: landscape,
        )),
        VerticalSeparator(),
        Flexible(child: GeneratedImageListView()),
      ],
    );
  }
}

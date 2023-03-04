import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/guidance_scale.dart';
import 'package:hard_diffusion/forms/fields/height.dart';
import 'package:hard_diffusion/forms/fields/inference_steps.dart';
import 'package:hard_diffusion/forms/fields/seed.dart';
import 'package:hard_diffusion/forms/fields/toggle_switch.dart';
import 'package:hard_diffusion/forms/fields/width.dart';

class AdvancedColumn extends StatelessWidget {
  const AdvancedColumn({
    super.key,
    required this.useRandomSeed,
    required this.useMultipleModels,
    required this.seed,
    required this.width,
    required this.height,
    required this.inferenceSteps,
    required this.guidanceScale,
    required this.setUseRandomSeed,
    required this.setUseMultipleModels,
    required this.setSeed,
    required this.setWidth,
    required this.setHeight,
    required this.setInferenceSteps,
    required this.setGuidanceScale,
  });

  final Function(bool) setUseRandomSeed;
  final Function(bool) setUseMultipleModels;
  final Function(int) setSeed;
  final Function(int) setWidth;
  final Function(int) setHeight;
  final Function(int) setInferenceSteps;
  final Function(double) setGuidanceScale;
  final bool useRandomSeed;
  final bool useMultipleModels;
  final int seed;
  final int width;
  final int height;
  final int inferenceSteps;
  final double guidanceScale;

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text("Advanced",
                  style: Theme.of(context).textTheme.titleLarge),
            ),
            ToggleSwitch(setValue: setUseRandomSeed, label: "Random Seed"),
            if (!useRandomSeed) ...[
              Text("Seed"),
              SeedField(
                value: seed,
                setValue: setSeed,
              ),
            ],
            Row(
              children: [
                WidthField(value: width, setValue: setWidth),
                HeightField(value: height, setValue: setHeight),
              ],
            ),
            Row(
              children: [
                InferenceStepsField(
                    value: inferenceSteps, setValue: setInferenceSteps),
                GuidanceScaleField(
                    value: guidanceScale, setValue: setGuidanceScale),
              ],
            ),
            Divider(),
            ToggleSwitch(
                setValue: setUseMultipleModels, label: "Use Multiple Models")
          ],
        ),
      ),
    );
  }
}

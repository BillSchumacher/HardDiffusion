import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/guidance_scale.dart';
import 'package:hard_diffusion/forms/fields/height.dart';
import 'package:hard_diffusion/forms/fields/inference_steps.dart';
import 'package:hard_diffusion/forms/fields/seed.dart';
import 'package:hard_diffusion/forms/fields/toggle_switch.dart';
import 'package:hard_diffusion/forms/fields/width.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:provider/provider.dart';

class AdvancedColumn extends StatelessWidget {
  const AdvancedColumn({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var useRandomSeed = appState.useRandomSeed;
    var seed = appState.seed;
    var width = appState.width;
    var height = appState.height;
    var inferenceSteps = appState.inferenceSteps;
    var guidanceScale = appState.guidanceScale;
    //var useMultipleModels = appState.useMultipleModels;

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
            ToggleSwitch(
                setValue: appState.setUseRandomSeed, label: "Random Seed"),
            if (!useRandomSeed) ...[
              Text("Seed"),
              SeedField(
                value: seed,
                setValue: appState.setSeed,
              ),
            ],
            Row(
              children: [
                WidthField(value: width, setValue: appState.setWidth),
                HeightField(value: height, setValue: appState.setHeight),
              ],
            ),
            Row(
              children: [
                InferenceStepsField(
                    value: inferenceSteps,
                    setValue: appState.setInferenceSteps),
                GuidanceScaleField(
                    value: guidanceScale, setValue: appState.setGuidanceScale),
              ],
            ),
            Divider(),
            ToggleSwitch(
                setValue: appState.setUseMultipleModels,
                label: "Use Multiple Models")
          ],
        ),
      ),
    );
  }
}

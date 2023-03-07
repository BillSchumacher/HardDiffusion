import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/float_input.dart';
import 'package:hard_diffusion/forms/fields/integer_input.dart';
import 'package:hard_diffusion/forms/fields/toggle_switch.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:provider/provider.dart';

class AdvancedColumn extends StatelessWidget {
  const AdvancedColumn({
    super.key,
    required this.landscape,
  });

  final bool landscape;

  @override
  Widget build(BuildContext context) {
    //var useMultipleModels = appState.useMultipleModels;
    var appState = context.watch<MyAppState>();
    var useRandomSeed = appState.useRandomSeed;
    var seed = appState.seed;
    var width = appState.width;
    var height = appState.height;
    var inferenceSteps = appState.inferenceSteps;
    var guidanceScale = appState.guidanceScale;

    if (landscape) {
      return Wrap(
        children: [
          Flexible(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text("Advanced",
                      style: Theme.of(context).textTheme.titleLarge),
                ),
              ],
            ),
          ),
          Flexible(
            child: Row(
                //padding: EdgeInsets.all(15.0),
                children: [
                  Flexible(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: ToggleSwitch(
                                  value: useRandomSeed,
                                  setValue: appState.setUseRandomSeed,
                                  label: "Random Seed"),
                            ),
                          ],
                        ),
                        if (!useRandomSeed) ...[
                          Row(
                            children: [
                              IntegerInputField(
                                label: "Seed",
                                value: seed,
                                setValue: appState.setSeed,
                              ),
                            ],
                          ),
                        ],
                        Row(
                          children: [
                            IntegerInputField(
                                label: "Width",
                                value: width,
                                setValue: appState.setWidth),
                            IntegerInputField(
                                label: "Height",
                                value: height,
                                setValue: appState.setHeight),
                          ],
                        ),
                        Row(
                          children: [
                            IntegerInputField(
                                label: "Inference Steps",
                                value: inferenceSteps,
                                setValue: appState.setInferenceSteps),
                            FloatInputField(
                                label: "Guidance Scale",
                                value: guidanceScale,
                                setValue: appState.setGuidanceScale),
                          ],
                        ),
                      ],
                    ),
                  ),
                  Flexible(
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: ToggleSwitch(
                                  value: appState.useMultipleModels,
                                  setValue: appState.setUseMultipleModels,
                                  label: "Use Multiple Models"),
                            ),
                          ],
                        ),
                        Row(
                          children: [
                            TextButton(
                              onPressed: () {
                                Navigator.pop(context);
                              },
                              child: const Text('Close'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  )
                ]),
          ),
        ],
      );
    }
    return Column(
      //padding: EdgeInsets.all(15.0),
      mainAxisAlignment: MainAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(8.0),
          child:
              Text("Advanced", style: Theme.of(context).textTheme.titleLarge),
        ),
        ToggleSwitch(
            value: useRandomSeed,
            setValue: appState.setUseRandomSeed,
            label: "Random Seed"),
        if (!useRandomSeed) ...[
          IntegerInputField(
            label: "Seed",
            value: seed,
            setValue: appState.setSeed,
          ),
        ],
        Row(
          children: [
            IntegerInputField(
                label: "Width", value: width, setValue: appState.setWidth),
            IntegerInputField(
                label: "Height", value: height, setValue: appState.setHeight),
          ],
        ),
        Row(
          children: [
            IntegerInputField(
                label: "Inference Steps",
                value: inferenceSteps,
                setValue: appState.setInferenceSteps),
            FloatInputField(
                label: "Guidance Scale",
                value: guidanceScale,
                setValue: appState.setGuidanceScale),
          ],
        ),
        Divider(),
        ToggleSwitch(
            value: appState.useMultipleModels,
            setValue: appState.setUseMultipleModels,
            label: "Use Multiple Models"),
        TextButton(
          onPressed: () {
            Navigator.pop(context);
          },
          child: const Text('Close'),
        ),
      ],
    );
  }
}

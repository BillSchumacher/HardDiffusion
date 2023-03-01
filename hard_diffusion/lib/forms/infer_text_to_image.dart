// Create a Form widget.
import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/advanced_switch.dart';
import 'package:hard_diffusion/forms/fields/guidance_scale.dart';
import 'package:hard_diffusion/forms/fields/height.dart';
import 'package:hard_diffusion/forms/fields/inference_steps.dart';
import 'package:hard_diffusion/forms/fields/nsfw_switch.dart';
import 'package:hard_diffusion/forms/fields/prompts.dart';
import 'package:hard_diffusion/forms/fields/random_seed_switch.dart';
import 'package:hard_diffusion/forms/fields/seed.dart';
import 'package:hard_diffusion/forms/fields/use_multiple_models_switch.dart';
import 'package:hard_diffusion/forms/fields/width.dart';
import 'package:hard_diffusion/vertical_separator.dart';

class InferTextToImageForm extends StatefulWidget {
  const InferTextToImageForm({super.key});

  @override
  InferTextToImageFormState createState() {
    return InferTextToImageFormState();
  }
}

class InferTextToImageFormState extends State<InferTextToImageForm> {
  final _formKey = GlobalKey<FormState>();
  String prompt = "";

  void setPrompt(value) {
    prompt = value;
  }

  String negativePrompt = "";

  void setNegativePrompt(value) {
    negativePrompt = value;
  }

  bool useMultipleModels = false;

  void setUseMultipleModels(value) {
    useMultipleModels = value;
  }

  bool useNsfw = false;

  void setUseNsfw(value) {
    useNsfw = value;
  }

  bool useAdvanced = true;

  void setUseAdvanced(value) {
    setState(() {
      useAdvanced = value;
    });
  }

  bool useRandomSeed = false;

  void setUseRandomSeed(value) {
    useRandomSeed = value;
  }

  int seed = 0;

  void setSeed(value) {
    seed = value;
  }

  int width = 512;

  void setWidth(value) {
    width = value;
  }

  int height = 512;

  void setHeight(value) {
    height = value;
  }

  int inferenceSteps = 50;

  void setInferenceSteps(value) {
    inferenceSteps = value;
  }

  double guidanceScale = 7.5;

  void setGuidanceScale(value) {
    guidanceScale = value;
  }

  void generate() {
    print("clicked generate!");
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();
    }
    print(_formKey.currentState.toString());
    print(_formKey);
    print(prompt);
    print(negativePrompt);
    print(useRandomSeed);
    print(width);
    print(height);
    print(inferenceSteps);
    print(guidanceScale);
    print(useAdvanced);
  }

  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.

    return Form(
      key: _formKey,
      child: Flexible(
        child: Row(children: [
          if (useAdvanced) ...[
            AdvancedColumn(
                useRandomSeed: useRandomSeed,
                seed: seed,
                width: width,
                height: height,
                inferenceSteps: inferenceSteps,
                guidanceScale: guidanceScale,
                setUseRandomSeed: setUseRandomSeed,
                setSeed: setSeed,
                setWidth: setWidth,
                setHeight: setHeight,
                setInferenceSteps: setInferenceSteps,
                setGuidanceScale: setGuidanceScale),
            VerticalSeparator(),
            PromptColumn(
              setPrompt: setPrompt,
              setNegativePrompt: setNegativePrompt,
              setUseAdvanced: setUseAdvanced,
              prompt: prompt,
              negativePrompt: negativePrompt,
              generate: generate,
            ),
          ] else ...[
            PromptColumn(
              setPrompt: setPrompt,
              setNegativePrompt: setNegativePrompt,
              setUseAdvanced: setUseAdvanced,
              prompt: prompt,
              negativePrompt: negativePrompt,
              generate: generate,
            ),
          ]
        ]),
      ),
    );
  }
}

class AdvancedColumn extends StatelessWidget {
  const AdvancedColumn({
    super.key,
    required this.useRandomSeed,
    required this.seed,
    required this.width,
    required this.height,
    required this.inferenceSteps,
    required this.guidanceScale,
    required this.setUseRandomSeed,
    required this.setSeed,
    required this.setWidth,
    required this.setHeight,
    required this.setInferenceSteps,
    required this.setGuidanceScale,
  });

  final Function(bool) setUseRandomSeed;
  final Function(int) setSeed;
  final Function(int) setWidth;
  final Function(int) setHeight;
  final Function(int) setInferenceSteps;
  final Function(double) setGuidanceScale;
  final bool useRandomSeed;
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
            Row(
              children: [
                Text("Seed"),
                RandomSeedSwitch(
                    value: useRandomSeed, setValue: setUseRandomSeed),
              ],
            ),
            SeedField(value: seed, setValue: setSeed),
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
            UseMultipleModelsSwitch() //value: useMultipleModels),
          ],
        ),
      ),
    );
  }
}

class PromptColumn extends StatelessWidget {
  const PromptColumn({
    super.key,
    required this.setPrompt,
    required this.setNegativePrompt,
    required this.setUseAdvanced,
    required this.generate,
    required this.prompt,
    required this.negativePrompt,
  });

  final Function(String) setPrompt;
  final Function(String) setNegativePrompt;
  final Function(bool) setUseAdvanced;
  final VoidCallback generate;
  final String prompt;
  final String negativePrompt;

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text("Generate",
                  style: Theme.of(context).textTheme.titleLarge),
            ),
            PromptField(promptValue: prompt, setPrompt: setPrompt),
            NegativePromptField(
                negativePromptValue: negativePrompt,
                setNegativePrompt: setNegativePrompt),
            Row(
              children: [
                AdvancedSwitch(setValue: setUseAdvanced),
                NSFWSwitch(),
              ],
            ),
            ElevatedButton(onPressed: generate, child: Text('Generate')),
          ],
        ),
      ),
    );
  }
}

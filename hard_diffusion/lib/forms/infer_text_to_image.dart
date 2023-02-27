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

// Create a corresponding State class.
// This class holds data related to the form.
class InferTextToImageFormState extends State<InferTextToImageForm> {
  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<InferTextToImageFormState>.
  final _formKey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.
    return Form(
      key: _formKey,
      child: Flexible(
        child: Row(
          children: [
            Flexible(
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
                        RandomSeedSwitch(),
                      ],
                    ),
                    SeedField(),
                    Row(
                      children: [
                        WidthField(),
                        HeightField(),
                      ],
                    ),
                    Row(
                      children: [
                        InferenceStepsField(),
                        GuidanceScaleField(),
                      ],
                    ),
                    Divider(),
                    UseMultipleModelsSwitch(),
                  ],
                ),
              ),
            ),
            VerticalSeparator(),
            Flexible(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text("Generate",
                          style: Theme.of(context).textTheme.titleLarge),
                    ),
                    PromptField(),
                    NegativePromptField(),
                    Row(
                      children: [
                        AdvancedSwitch(),
                        NSFWSwitch(),
                      ],
                    ),
                    ElevatedButton(onPressed: null, child: Text('Generate')),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

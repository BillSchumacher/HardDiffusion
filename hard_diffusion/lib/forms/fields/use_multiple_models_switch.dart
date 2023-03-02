import 'package:flutter/material.dart';

class UseMultipleModelsSwitch extends StatefulWidget {
  const UseMultipleModelsSwitch({super.key});

  @override
  State<UseMultipleModelsSwitch> createState() =>
      _UseMultipleModelsSwitchState();
}

class _UseMultipleModelsSwitchState extends State<UseMultipleModelsSwitch> {
  bool light = true;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text('Multiple Models:'),
        Switch(
          // This bool value toggles the switch.
          value: light,
          activeColor: Colors.orange,
          onChanged: (bool value) {
            // This is called when the user toggles the switch.
            setState(() {
              light = value;
            });
          },
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';

class AdvancedSwitch extends StatefulWidget {
  const AdvancedSwitch({super.key, required this.setValue});

  final Function(bool) setValue;
  @override
  State<AdvancedSwitch> createState() => _AdvancedSwitchState(
        setValue: setValue,
      );
}

class _AdvancedSwitchState extends State<AdvancedSwitch> {
  _AdvancedSwitchState({required this.setValue});
  bool light = true;

  Function(bool) setValue;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text('Advanced:'),
        Switch(
          // This bool value toggles the switch.
          value: light,
          activeColor: Colors.orange,
          onChanged: (bool value) {
            // This is called when the user toggles the switch.
            setState(() {
              light = value;
            });
            setValue(light);
          },
        ),
      ],
    );
  }
}

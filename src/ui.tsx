import { Pressable, StyleSheet, Text } from 'react-native';

// Matches the Vetanly dashboard.
export const colors = {
  brand: '#5938E8',
  brandSoft: '#EEEAFE',
  success: '#16A34A',
  warning: '#EA580C',
  danger: '#DC2626',
  text: '#111827',
  muted: '#6B7280',
  border: '#E5E7EB',
};

export const errorMessage = (e: unknown) =>
  e instanceof TypeError ? 'Could not reach the server' : e instanceof Error ? e.message : String(e);

export function Button({
  label,
  onPress,
  disabled,
  secondary,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  secondary?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled }}
      disabled={disabled}
      onPress={onPress}
      style={[ui.button, secondary && ui.buttonSecondary, disabled && ui.disabled]}
    >
      <Text style={[ui.buttonText, secondary && ui.buttonTextSecondary]}>{label}</Text>
    </Pressable>
  );
}

/** White panels shown over the camera for manager-only screens. */
export const ui = StyleSheet.create({
  panel: {
    position: 'absolute',
    alignSelf: 'center',
    top: '10%',
    width: '90%',
    maxWidth: 560,
    backgroundColor: '#fff',
    borderRadius: 24,
    padding: 24,
    gap: 16,
    elevation: 12,
  },
  tall: { bottom: '6%', top: '6%' },
  title: { fontSize: 24, fontWeight: '700', color: colors.text },
  hint: { fontSize: 15, color: colors.muted, lineHeight: 21 },
  error: { color: colors.danger, fontSize: 15, fontWeight: '500' },
  row: { flexDirection: 'row', gap: 12, justifyContent: 'flex-end' },
  button: { backgroundColor: colors.brand, borderRadius: 14, paddingVertical: 14, paddingHorizontal: 24, alignItems: 'center' },
  buttonSecondary: { backgroundColor: colors.brandSoft },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '700' },
  buttonTextSecondary: { color: colors.brand },
  disabled: { opacity: 0.5 },
});

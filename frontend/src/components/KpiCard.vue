<template>
  <section class="kpi-card">
    <span>{{ label }}</span>
    <strong>{{ value }}</strong>
    <small v-if="hint">{{ hint }}</small>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  label: string;
  value: string;
  hint?: string;
}>();
</script>


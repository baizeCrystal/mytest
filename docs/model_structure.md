# Student Action Error Model

## Overall Architecture

```mermaid
flowchart LR
    A["Student sports video<br/>frame folder"] --> B["StudentActionDataset<br/>sample T RGB frames"]
    B --> C["Temporal Backbone<br/>X3D / Action-Slot / SimpleCNN"]
    C --> D["Temporal Features<br/>F: B x T' x C"]

    D --> E["SoftPhaseAssignment<br/>learnable phase queries"]
    E --> F["Phase Weights<br/>A: B x K x T'"]
    E --> G["Student Phase Features<br/>S: B x K x C"]

    G --> K["Variant A: Prototype contrast<br/>[S, P, |S-P|, S*P]"]
    H["Action ID"] --> I["CorrectActionPrototypeBank"]
    I --> K

    D --> L["Variant B: PhaseAwarePartSlotAggregator<br/>phase mask x spatial feature map"]
    F --> L
    L --> M["Part-slot Phase Context<br/>[S, Slot, |S-Slot|]<br/>B x K x 3C"]

    K --> N["Phase Error Head"]
    M --> N
    N --> O["Phase Error Logits<br/>B x K x E"]

    K --> P["Flatten K phases"]
    M --> P
    P --> Q["Video Error Head"]
    Q --> R["Video Error Logits<br/>B x E"]

    R --> S["BCE Loss<br/>video-level error labels"]
    O --> T["BCE Loss<br/>phase-level auxiliary logits"]
    F --> U["Optional phase supervision<br/>if phase_labels are provided"]
```

## Soft Phase Assignment

```mermaid
flowchart TB
    A["Temporal features<br/>F: B x T' x C"] --> B["LayerNorm"]
    C["Learnable phase queries<br/>Q: K x C"] --> D["LayerNorm"]
    B --> E["Dot product<br/>Q · F"]
    D --> E
    E --> F["Softmax over time"]
    F --> G["Phase weights<br/>A: B x K x T'"]
    G --> H["Weighted temporal pooling"]
    A --> H
    H --> I["Phase features<br/>S: B x K x C"]
```

Formula:

```text
A_phase = softmax((Q_phase * F_t) / sqrt(C), dim=time)
S_k = sum_t A_phase[k, t] * F_t
```

## Prototype Variant

```mermaid
flowchart LR
    A["Correct-action samples<br/>is_correct=true"] --> B["Same backbone"]
    B --> C["SoftPhaseAssignment"]
    C --> D["Correct phase features<br/>B x K x C"]
    D --> E["Group by action_id"]
    E --> F["Mean over samples"]
    F --> G["Prototype tensor<br/>num_actions x K x C"]
    G --> H["correct_prototypes.pth"]
```

## Runtime Tensor Shapes

| Symbol | Meaning | Shape |
| --- | --- | --- |
| `videos` | sampled RGB clip as list of tensors | `T x [B, 3, H, W]` |
| `F` | temporal features | `B x T' x C` |
| `A_phase` | soft phase assignment weights | `B x K x T'` |
| `S` | student phase features | `B x K x C` |
| `P` | correct phase prototypes selected by `action_id` | `B x K x C` |
| `X_spatial` | spatiotemporal feature map | `B x T' x H' x W' x C` |
| `Z_part` | phase-aware part-slot tokens | `B x K x P x C` |
| `D` | prototype contrast tensor `[S, P, abs(S-P), S*P]` | `B x K x 4C` |
| `D_part` | part-slot context tensor `[S, Slot, abs(S-Slot)]` | `B x K x 3C` |
| `phase_logits` | phase-wise error logits | `B x K x E` |
| `logits` | video-level error logits | `B x E` |

Where:

- `B`: batch size
- `T`: input frame count
- `T'`: backbone temporal output length
- `C`: feature dimension
- `K`: number of soft action phases
- `E`: number of error classes
- `P`: number of body-part slots

## Patent-Oriented Module Naming

```text
Student video acquisition module
  -> Action-slot temporal feature extraction module
  -> Soft action phase assignment module
  -> Optional correct action prototype knowledge base
  -> Optional phase-aware human-part slot aggregation module
  -> Student action error recognition module
```

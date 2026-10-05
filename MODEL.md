# Exact model and interpretation

## Stochastic game

There are N=24 agents in four information-delivery groups. Each needs eight units of service and chooses activate (one unit) or wait at each of 42 rounds. An agent with no remaining service must wait. Agents interact through total active count D, not opponent identities. Group membership controls delivery timing; it does not partition physical interaction. Costs are structurally symmetric with small heterogeneous holding preferences; the exact-symmetry control sets these differences to zero.

Public physical state x is an integer from 0 to 4. Conditional on D, the next state is stochastic rounding of .65 x + .35(4 D/N), mixed with probability .08 with a uniform draw over the five states. Thus usage affects future physical conditions. Quantization and exogenous randomness prevent direct inversion of state transitions to recover D.

Before acting, agent i has remaining service r_i. Its stage cost is

`(.01 + .5 * indicator[t >= 30]) * h_i * r_i + a_i * (.15 + 3 D/N + x/4)`.

Here h_i=exp(.03 Z_i), Z_i independently standard normal. Terminal remaining service costs 8 per unit. The learner uses discount .98. Reported realized total costs are undiscounted, including terminal cost. This concrete test-bed cost is a modeling choice for reproducibility, not a proposed restriction on the grant's research program. Physical pressure is a generic shared resource condition, not a calibrated feeder or queue model. No potential-game property is established for this dynamic game.

## Who observes what

Agents know the cost structure, physical transition model and their own holding factor. They observe their own service/action history and the public physical state. They do not observe other agents' actions or backlog directly. **The realized congestion-dependent bill is settled after the episode**, so it does not reveal total usage during adaptation; allowing agents to observe an invertible per-round bill would defeat this information design and would require a different experiment.

The designer observes aggregate usage and sends truthful cumulative historical usage records. An agent subtracts its own historical contribution to obtain an exact opposing count for each disclosed round. There are no fabricated forecasts, personalized prices or changed utilities. The intervention is delivery timing, one admissible form of information policy. This study does not establish equivalent results for price forecasts or other signal semantics.

Under common periodic delivery, everyone receives at the same phase of a six-round period. Fixed staggering uses phases (0,1,3,4) across four equal groups. Common randomized delivery independently draws one phase per agent at episode start from the same uniform distribution; phases then stay fixed. Each receives five updates during rounds 1–30. Starting at round 31, these policies and the full-information benchmark receive every round. The public-only benchmark never receives designer records. Equal update counts do not imply equal mutual information or communication bytes: different timing and endogenous trajectories change informative content.

## Imperfect stochastic fictitious-play-style learner

Each agent maintains an empirical distribution over integer opposing active counts k=0,...,N-1, separately by public source state. Each state's initial distribution is Binomial(N-1,.4), with pseudocount strength 3. The nominal learner retains all records. Memory mismatch retains only records from the last M rounds, with M in {4,8,16,24}; age is measured from the event, not the disclosure.

After each public state transition, the learner computes a posterior opposing-count distribution using its preceding belief and the known transition likelihood conditional on its own action. This contributes one soft empirical record. A later exact designer disclosure replaces the soft record; it does not add a second observation. These are approximate empirical learning dynamics, not an exact Bayesian filter of all joint agent histories.

At each decision, backward dynamic programming computes finite-horizon Q values assuming the current state-conditioned opposing-count distribution remains frozen for planning. The recursion includes remaining service and public physical state, but does not model future belief updates or opponents' individual backlogs. Activation probability is logistic((Q_wait-Q_activate)/.025), with independent private random draws. This is an explicitly specified imperfect stochastic-FP-style rule, not a claim that a standard stochastic FP convergence theorem applies.

The hypothesized chain is delivery -> changes in empirical beliefs -> aligned action-value crossings -> simultaneous activation -> shared physical response -> new observations. Recorded belief/gap/action/state trajectories make the chain inspectable. Randomized and staggered timing intervene on delivery while holding preferences and exogenous randomness fixed. These comparisons support timing's causal effect within this simulator; they do not separately prove every link or establish a universal reinforcement mechanism.

## Metrics

Let d_t=D_t/N and set d_-1=0. Let f_t be the corresponding usage under the matched frozen-belief reference. The excess-buildup score is

`H = max(0, max over t and 1<=ell<=6 of [(d_t-d_(t-ell))-(f_t-f_(t-ell))])`,

using valid indices within the episode. H can exceed 1 (upper bound 2), is sensitive to relative wave timing, and is zero by definition for the frozen reference. It is not a complete definition of harmful herding. Always interpret it with peak usage, binary actions, service, costs and reference decision loss. Raw buildup and switching/alignment statistics are also saved.

The full-history reference decision loss uses the complete true opposing-count history along the *same visited trajectory*, with the same memory window. It computes the expected Q loss of the policy's mixed action relative to the minimum reference Q value. We average over all 42 rounds for each agent and take the worst agent per episode. The reported mean averages that quantity across independent seeds. This is an empirical one-decision planning reference, not regret, a Nash gap, welfare loss or actual-cost optimality. Softmax randomness can give positive loss even with full information.

Deadline service is the fraction completed after actions 0–29. Final service is completion after all 42 actions. Tail peak includes rounds 30–41; common full-information follow-up begins at 31. Participation is the fraction switching each round; directional alignment is absolute net switching divided by total switching, zero when no one switches.

Differentiation is the mean cyclic phase distance to the nearest common phase, divided by six, for deterministic phase policies. Common randomized timing has differentiation zero because all agents receive the same conditional policy distribution, despite differing realized phases. This is a menu-specific measure, not a universal distance between information policies.

## Controls and scope

Controls vary exact symmetry, log-holding-factor standard deviation .15, zero state-dependent cost, omission of public-state inference, and N=48. The omitted-inference control has no soft records until exact records arrive; it deliberately changes the learner's use of available information. Zero pressure cost removes only the state-to-cost term, not all feedback. Shared service deadlines and initial backlogs remain a source of symmetry/synchrony and are not varied here. Other limits include the one-resource game, short horizon, known transition model and finite policy menu. No analytical no-herding certificate, real-infrastructure validation or reliable feedback recovery is claimed.

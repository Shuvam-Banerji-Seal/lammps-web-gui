import type { Pair } from '../force/types';
import { PairFepSoft, PairLJCharmmSoftCoulLong, PairSoftUnsupported } from '../force/pair/fep_soft';

/** soft-core FEP pair styles (pair_fep_soft) (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/soft': () => new PairFepSoft('lj/cut/soft', 'lj12', 'none'),
  'lj/cut/coul/cut/soft': () => new PairFepSoft('lj/cut/coul/cut/soft', 'lj12', 'cut'),
  'lj/cut/coul/long/soft': () => new PairFepSoft('lj/cut/coul/long/soft', 'lj12', 'long'),
  'coul/cut/soft': () => new PairFepSoft('coul/cut/soft', 'none', 'cut'),
  'coul/long/soft': () => new PairFepSoft('coul/long/soft', 'none', 'long'),
  'lj/class2/soft': () => new PairFepSoft('lj/class2/soft', 'class2', 'none'),
  'lj/class2/coul/cut/soft': () => new PairFepSoft('lj/class2/coul/cut/soft', 'class2', 'cut'),
  'lj/class2/coul/long/soft': () => new PairFepSoft('lj/class2/coul/long/soft', 'class2', 'long'),
  'lj/charmm/coul/long/soft': () => new PairLJCharmmSoftCoulLong(),
  'lj/cut/tip4p/long/soft': () => new PairSoftUnsupported('lj/cut/tip4p/long/soft', 'the TIP4P massless-site pair machinery is not reused for the soft styles'),
  'tip4p/long/soft': () => new PairSoftUnsupported('tip4p/long/soft', 'the TIP4P massless-site pair machinery is not reused for the soft styles'),
  'morse/soft': () => new PairSoftUnsupported('morse/soft', 'the piecewise soft Morse form is not implemented'),
};
